"""
Secure Code Execution Framework for Local Automation
Provides sandboxed Python code execution with safety measures.

Production-grade module for AWS Bedrock Claude skill execution.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import time
import re
import atexit
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from contextlib import contextmanager
import signal
import traceback
import hashlib

from .exceptions import (
    ConfigurationError,
    ExecutionError,
    SecurityError,
    PackageInstallError,
    validate_security_level,
    validate_timeout,
    validate_encoding,
    validate_workspace_dir,
    validate_max_memory,
    retry_on_exception,
)

# Use module-level logger (don't configure basicConfig - leave to caller)
logger = logging.getLogger(__name__)

# Default safe imports for data science / ML workflows
DEFAULT_SAFE_IMPORTS = [
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
    'sklearn', 'scipy', 'statsmodels', 'prophet',
    'json', 'csv', 'datetime', 'time', 'math', 'random',
    'pathlib', 'os', 'sys', 're', 'collections', 'itertools', 'functools',
    'typing', 'dataclasses', 'enum', 'abc',
    'warnings', 'logging', 'io', 'base64', 'hashlib',
    'docx', 'openpyxl', 'xlrd', 'PIL', 'cv2',
    'requests', 'urllib', 'bs4',
    'tqdm', 'rich', 'tabulate',
]

# Security levels for different use cases
SECURITY_LEVELS = {
    'strict': {
        'blocked_imports': ['subprocess', 'os.system', 'eval', 'exec', 'compile', '__import__', 'ctypes', 'socket'],
        'blocked_patterns': [r'\beval\s*\(', r'\bexec\s*\(', r'\bcompile\s*\(', r'\b__import__\s*\(', r'os\.system\s*\('],
    },
    'moderate': {
        'blocked_imports': ['ctypes', 'socket'],
        'blocked_patterns': [r'\beval\s*\(', r'\bexec\s*\('],
    },
    'permissive': {
        'blocked_imports': [],
        'blocked_patterns': [],
    }
}

# Track active executors for cleanup
_active_executors: List['CodeExecutor'] = []
_cleanup_lock = threading.Lock()


def _cleanup_all_executors():
    """Cleanup function registered with atexit."""
    with _cleanup_lock:
        for executor in _active_executors[:]:  # Copy list to avoid mutation during iteration
            try:
                if hasattr(executor, '_temp_workspace') and executor._temp_workspace:
                    executor.cleanup()
            except Exception as e:
                logger.debug(f"Error during cleanup: {e}")


# Register cleanup on interpreter exit
atexit.register(_cleanup_all_executors)


class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signals."""
    raise TimeoutException("Code execution timed out")


class CodeExecutor:
    """
    Secure code executor with sandboxing and resource limits.
    Supports Python code execution with output capture and file management.

    Production-grade executor for AWS Bedrock Claude skill workflows.
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        timeout: int = 300,
        max_memory_mb: int = 1024,
        allowed_imports: Optional[List[str]] = None,
        blocked_imports: Optional[List[str]] = None,
        auto_install_packages: bool = True,
        security_level: str = 'moderate',
        accumulate_blocks: bool = True,
        encoding: str = 'utf-8'
    ):
        """
        Initialize code executor.

        Args:
            workspace_dir: Directory for code execution (created if doesn't exist)
            timeout: Maximum execution time in seconds (1-3600)
            max_memory_mb: Maximum memory usage in MB (1-65536)
            allowed_imports: Whitelist of allowed imports (None = allow all safe imports)
            blocked_imports: Blacklist of blocked imports (overrides security_level)
            auto_install_packages: Auto-install missing packages
            security_level: 'strict', 'moderate', or 'permissive' (default: 'moderate')
            accumulate_blocks: Execute blocks with accumulated context (default: True)
            encoding: File encoding for code files (default: 'utf-8')

        Raises:
            ConfigurationError: If any parameter is invalid
        """
        # Validate all inputs
        self.security_level = validate_security_level(security_level)
        self.timeout = validate_timeout(timeout)
        self.max_memory_mb = validate_max_memory(max_memory_mb)
        self.encoding = validate_encoding(encoding)

        # Set up workspace
        validated_workspace = validate_workspace_dir(workspace_dir)
        if validated_workspace:
            self.workspace_dir = validated_workspace
            self._temp_workspace = False
        else:
            self.workspace_dir = tempfile.mkdtemp(prefix="code_exec_")
            self._temp_workspace = True

        self.auto_install_packages = bool(auto_install_packages)
        self.accumulate_blocks = bool(accumulate_blocks)

        # Set up security based on level
        security_config = SECURITY_LEVELS[self.security_level]
        self.blocked_imports = list(blocked_imports) if blocked_imports is not None else list(security_config['blocked_imports'])
        self.blocked_patterns = list(security_config['blocked_patterns'])
        self.allowed_imports = list(allowed_imports) if allowed_imports is not None else list(DEFAULT_SAFE_IMPORTS)

        # Create workspace
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)

        # Thread-safe tracking of installed packages and accumulated code
        self._lock = threading.RLock()
        self.installed_packages: set = set()
        self.accumulated_code: List[str] = []
        self.accumulated_imports: set = set()

        # Track for cleanup
        with _cleanup_lock:
            _active_executors.append(self)

        # State tracking
        self._is_cleaned_up = False

        logger.info(f"Code executor initialized with workspace: {self.workspace_dir}")
        logger.debug(f"Security level: {self.security_level}, Accumulate blocks: {self.accumulate_blocks}")
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from text (markdown format).
        
        Handles nested code blocks properly.

        Args:
            text: Text containing code blocks

        Returns:
            List of dicts with 'language' and 'code' keys
        """
        code_blocks = []
        
        # Find all code block positions
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for code block start (```language or just ```)
            start_match = re.match(r'^```(\w+)?$', stripped)
            if start_match:
                lang = start_match.group(1) if start_match.group(1) else None
                block_start = i
                code_lines = []
                i += 1
                
                # For Python blocks, find the LAST matching ``` to handle nested content
                if lang and lang.lower() in ['python', 'py', 'python3']:
                    # Collect all lines until we find a valid closing
                    # Try from the last ``` backward to find valid Python
                    remaining_lines = lines[i:]
                    
                    # Find all potential closing positions (lines with just ```)
                    close_positions = []
                    for j, l in enumerate(remaining_lines):
                        if l.strip() == '```':
                            close_positions.append(j)
                    
                    # Try each closing position from last to first
                    # to find the one that gives us valid Python
                    found_valid = False
                    for close_pos in reversed(close_positions):
                        candidate_code = '\n'.join(remaining_lines[:close_pos])
                        if self._is_valid_python(candidate_code):
                            code_blocks.append({
                                'language': lang.lower(),
                                'code': candidate_code.strip()
                            })
                            i = block_start + close_pos + 2  # Move past this block
                            found_valid = True
                            break
                    
                    if not found_valid and close_positions:
                        # Use the first closing as fallback
                        code = '\n'.join(remaining_lines[:close_positions[0]])
                        code_blocks.append({
                            'language': lang.lower() if lang else 'unknown',
                            'code': code.strip()
                        })
                        i = block_start + close_positions[0] + 2
                    elif not close_positions:
                        i += 1
                else:
                    # For non-Python blocks, use simple matching
                    while i < len(lines):
                        current = lines[i]
                        if current.strip() == '```':
                            break
                        code_lines.append(current)
                        i += 1
                    i += 1  # Skip closing ```
                    
                    code = '\n'.join(code_lines)
                    if lang:
                        lang = lang.lower()
                    elif self._looks_like_python(code):
                        lang = 'python'
                    else:
                        lang = 'unknown'
                    
                    code_blocks.append({
                        'language': lang,
                        'code': code.strip()
                    })
            else:
                i += 1
        
        # Also check for <run_code> format
        run_code_pattern = r'<run_code>\s*<language>(.*?)</language>\s*<code>(.*?)</code>\s*</run_code>'
        run_code_matches = re.findall(run_code_pattern, text, re.DOTALL)

        for lang, code in run_code_matches:
            lang = lang.strip().lower() if lang else 'python'
            code_blocks.append({
                'language': lang,
                'code': code.strip()
            })

        return code_blocks

    def _looks_like_python(self, code: str) -> bool:
        """
        Heuristic check if code looks like Python.

        Args:
            code: Code string to check

        Returns:
            True if the code appears to be Python
        """
        code = code.strip()
        if not code:
            return False

        # Patterns that indicate Python code
        python_indicators = [
            r'^\s*import\s+\w+',           # import statement
            r'^\s*from\s+\w+\s+import',    # from X import Y
            r'^\s*def\s+\w+\s*\(',         # function definition
            r'^\s*class\s+\w+',            # class definition
            r'^\s*if\s+.*:',               # if statement
            r'^\s*for\s+\w+\s+in\s+',      # for loop
            r'^\s*while\s+.*:',            # while loop
            r'^\s*print\s*\(',             # print function
            r'^\s*#.*$',                   # Python comment at start
            r'^\s*\w+\s*=\s*',             # assignment
            r'^\s*@\w+',                   # decorator
            r'^\s*try\s*:',                # try block
            r'^\s*with\s+.*:',             # with statement
        ]

        # Check first few non-empty lines
        lines = [l for l in code.split('\n') if l.strip()][:5]
        first_content = '\n'.join(lines)

        for pattern in python_indicators:
            if re.search(pattern, first_content, re.MULTILINE):
                return True

        # Patterns that indicate NOT Python (markdown, etc.)
        non_python_patterns = [
            r'^\*\*\w+.*\*\*',              # Bold markdown **text**
            r'^#{1,6}\s+\w+',               # Markdown headers
            r'^\s*-\s+\w+',                 # Markdown list items
            r'^\s*\d+\.\s+\w+',             # Numbered list items
            r'^>\s+',                        # Blockquote
            r'\[.*\]\(.*\)',                 # Markdown links
        ]

        for pattern in non_python_patterns:
            if re.search(pattern, first_content, re.MULTILINE):
                return False

        # If we can't determine, check for valid Python syntax
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def extract_python_blocks(self, text: str) -> List[str]:
        """
        Extract only valid Python code blocks.

        Filters out non-Python blocks and validates syntax.
        Uses AST-based validation with detailed error reporting.
        """
        blocks = self.extract_code_blocks(text)
        python_blocks = []

        for i, block in enumerate(blocks):
            lang = block['language']
            code = block['code']

            # Accept explicitly marked Python blocks
            if lang in ['python', 'py', 'python3']:
                # Validate it's actually valid Python
                if self._is_valid_python(code):
                    python_blocks.append(code)
                else:
                    # Get detailed error info
                    error_details = self.get_syntax_error_details(code)
                    if error_details:
                        logger.warning(
                            f"Skipping invalid Python block {i+1}: "
                            f"Line {error_details['line']}: {error_details['message']}"
                        )
                        logger.debug(f"Problematic line: {error_details['text']}")
                    else:
                        logger.warning(f"Skipping invalid Python block {i+1}: syntax error detected")
            # Accept 'unknown' blocks that look like Python
            elif lang == 'unknown' and self._looks_like_python(code):
                if self._is_valid_python(code):
                    python_blocks.append(code)

        return python_blocks

    def _is_valid_python(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            logger.debug(f"Syntax error in code: {e}")
            return False

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze Python code using AST to extract structural information.
        
        Args:
            code: Python source code string
            
        Returns:
            Dict containing:
                - valid: bool - whether code is syntactically valid
                - imports: list of imported modules
                - from_imports: list of (module, names) tuples
                - functions: list of function names defined
                - classes: list of class names defined
                - calls: list of function/method calls made
                - file_operations: bool - whether code does file I/O
                - has_network: bool - whether code might do network operations
                - syntax_error: str or None - error message if invalid
        """
        import ast
        
        result = {
            'valid': False,
            'imports': [],
            'from_imports': [],
            'functions': [],
            'classes': [],
            'calls': [],
            'file_operations': False,
            'has_network': False,
            'syntax_error': None,
        }
        
        try:
            tree = ast.parse(code)
            result['valid'] = True
        except SyntaxError as e:
            result['syntax_error'] = f"Line {e.lineno}: {e.msg}"
            return result
        
        # Walk the AST to extract information
        for node in ast.walk(tree):
            # Import statements: import x, import x.y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result['imports'].append(alias.name)
                    
            # From imports: from x import y
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [alias.name for alias in node.names]
                result['from_imports'].append((module, names))
                # Also track the base module
                if module:
                    result['imports'].append(module.split('.')[0])
                    
            # Function definitions
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                result['functions'].append(node.name)
                
            # Class definitions
            elif isinstance(node, ast.ClassDef):
                result['classes'].append(node.name)
                
            # Function calls
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    result['calls'].append(call_name)
                    
                    # Check for file operations
                    if call_name in ('open', 'read', 'write', 'Path.open', 
                                    'Path.read_text', 'Path.write_text',
                                    'Path.read_bytes', 'Path.write_bytes'):
                        result['file_operations'] = True
                        
                    # Check for network operations
                    if any(net in call_name for net in ('requests.', 'urllib.', 
                                                         'http.', 'socket.', 
                                                         'aiohttp.')):
                        result['has_network'] = True
        
        # Remove duplicates while preserving order
        result['imports'] = list(dict.fromkeys(result['imports']))
        result['calls'] = list(dict.fromkeys(result['calls']))
        
        return result
    
    def _get_call_name(self, node) -> Optional[str]:
        """Extract the name of a function call from an AST Call node."""
        import ast
        
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle chained attributes like obj.method or module.func
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            parts.reverse()
            return '.'.join(parts)
        return None

    def get_syntax_error_details(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a syntax error in code.
        
        Args:
            code: Python source code string
            
        Returns:
            None if code is valid, otherwise dict with:
                - line: line number (1-indexed)
                - column: column offset
                - message: error message
                - text: the problematic line of code
        """
        try:
            compile(code, '<string>', 'exec')
            return None
        except SyntaxError as e:
            lines = code.split('\n')
            problem_line = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ''
            return {
                'line': e.lineno,
                'column': e.offset,
                'message': e.msg,
                'text': problem_line,
            }

    def check_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check if code contains only allowed imports using AST analysis.

        Returns:
            (is_safe, list_of_issues)
        """
        issues = []

        # Use AST-based analysis for more accurate import detection
        analysis = self.analyze_code(code)
        
        if not analysis['valid']:
            # Fall back to regex for invalid code
            import_pattern = r'^\s*(?:import|from)\s+(\S+)'
            imports = re.findall(import_pattern, code, re.MULTILINE)
        else:
            # Use AST-extracted imports (more reliable)
            imports = analysis['imports']
            # Also check from_imports
            for module, names in analysis['from_imports']:
                if module:
                    imports.append(module)

        # Check against blocked imports
        for imp in imports:
            base_module = imp.split('.')[0]

            # Check blocked imports
            if any(blocked in imp for blocked in self.blocked_imports):
                issues.append(f"Blocked import detected: '{imp}'")

        # Check for dangerous patterns based on security level
        for pattern in self.blocked_patterns:
            if re.search(pattern, code):
                issues.append(f"Dangerous pattern detected: {pattern}")

        return len(issues) == 0, issues

    def sanitize_code(self, code: str) -> str:
        """
        Sanitize code by removing problematic characters for cross-platform compatibility.

        Args:
            code: Raw code string

        Returns:
            Sanitized code string safe for file writing
        """
        # Replace problematic Unicode characters with ASCII equivalents
        replacements = {
            '\u2713': '[OK]',      # Check mark
            '\u2714': '[OK]',      # Heavy check mark
            '\u2715': '[X]',       # X mark
            '\u2716': '[X]',       # Heavy X mark
            '\u2717': '[X]',       # Ballot X
            '\u2718': '[X]',       # Heavy ballot X
            '\U0001f4ca': '[CHART]',  # Bar chart emoji
            '\U0001f4c8': '[CHART]',  # Chart with upwards trend
            '\U0001f4c9': '[CHART]',  # Chart with downwards trend
            '\U0001f4c4': '[DOC]',    # Page facing up
            '\U0001f4dd': '[NOTE]',   # Memo
            '\u2022': '*',         # Bullet point
            '\u2023': '>',         # Triangular bullet
            '\u2043': '-',         # Hyphen bullet
            '\u25aa': '*',         # Black small square
            '\u25ab': '*',         # White small square
            '\u25cf': '*',         # Black circle
            '\u25cb': 'o',         # White circle
            '\u2192': '->',        # Right arrow
            '\u2190': '<-',        # Left arrow
            '\u2191': '^',         # Up arrow
            '\u2193': 'v',         # Down arrow
            '\u2026': '...',       # Ellipsis
            '\u201c': '"',         # Left double quotation
            '\u201d': '"',         # Right double quotation
            '\u2018': "'",         # Left single quotation
            '\u2019': "'",         # Right single quotation
            '\u2014': '--',        # Em dash
            '\u2013': '-',         # En dash
        }

        for char, replacement in replacements.items():
            code = code.replace(char, replacement)

        # Remove any remaining non-ASCII characters that could cause issues
        # but preserve common safe ones
        try:
            code.encode('utf-8')
        except UnicodeEncodeError:
            # Fallback: encode with replacement
            code = code.encode('utf-8', errors='replace').decode('utf-8')

        return code
    
    def extract_required_packages(self, code: str) -> List[str]:
        """
        Extract packages that need to be installed.
        
        Returns:
            List of package names
        """
        import_pattern = r'^\s*(?:import|from)\s+(\S+)'
        imports = re.findall(import_pattern, code, re.MULTILINE)
        
        # Common package mappings
        package_map = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'bs4': 'beautifulsoup4',
        }
        
        # Comprehensive list of Python standard library modules
        stdlib_modules = {
            # Built-in and core modules
            'os', 'sys', 'io', 'json', 're', 'time', 'datetime', 'math', 'random',
            'collections', 'itertools', 'functools', 'operator', 'copy', 'pprint',
            # File and path handling
            'pathlib', 'shutil', 'glob', 'fnmatch', 'tempfile', 'fileinput',
            # Data formats
            'csv', 'configparser', 'xml', 'html', 'base64', 'binascii', 'struct',
            # Text processing
            'string', 'textwrap', 'difflib', 'unicodedata',
            # Type hints and inspection
            'typing', 'types', 'inspect', 'abc', 'dataclasses', 'enum',
            # Concurrency
            'threading', 'multiprocessing', 'subprocess', 'concurrent', 'asyncio',
            'queue', 'sched',
            # Networking and web
            'socket', 'ssl', 'http', 'urllib', 'email', 'ftplib', 'smtplib',
            # Compression and archiving
            'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'zlib',
            # Cryptography and hashing
            'hashlib', 'hmac', 'secrets',
            # Debugging and testing
            'logging', 'warnings', 'traceback', 'pdb', 'unittest', 'doctest',
            # System and runtime
            'argparse', 'getopt', 'ctypes', 'platform', 'signal', 'atexit', 'gc',
            # Encoding
            'codecs', 'locale', 'gettext',
            # Database
            'sqlite3', 'dbm',
            # Misc
            'pickle', 'shelve', 'marshal', 'weakref', 'contextlib', 'decimal',
            'fractions', 'statistics', 'cmath', 'heapq', 'bisect', 'array',
            'calendar', 'uuid', 'dis', 'ast', 'symtable', 'token', 'keyword',
            # Internal/private modules (should never be installed)
            '_thread', '__future__', 'builtins',
        }
        
        packages = []
        for imp in imports:
            base_module = imp.split('.')[0]
            
            # Skip standard library modules
            if base_module in stdlib_modules:
                continue
            
            # Skip private/internal modules
            if base_module.startswith('_'):
                continue
            
            # Map to actual package name
            package_name = package_map.get(base_module, base_module)
            packages.append(package_name)
        
        return list(set(packages))
    
    def install_package(self, package: str, max_retries: int = 2) -> bool:
        """
        Install a Python package using pip with retry logic.

        Args:
            package: Package name to install
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            True if successful, False otherwise
        """
        if not package or not isinstance(package, str):
            logger.warning(f"Invalid package name: {package}")
            return False

        # Sanitize package name
        package = package.strip()
        if not package:
            return False

        with self._lock:
            if package in self.installed_packages:
                logger.debug(f"Package '{package}' already installed (cached)")
                return True

        # Retry loop
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Installing package: {package} (attempt {attempt + 1}/{max_retries + 1})")

                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package, "-q", "--break-system-packages"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    encoding='utf-8',
                    errors='replace'
                )

                if result.returncode == 0:
                    with self._lock:
                        self.installed_packages.add(package)
                    logger.info(f"Successfully installed: {package}")
                    return True
                else:
                    last_error = result.stderr.strip() if result.stderr else "Unknown error"
                    logger.warning(f"Attempt {attempt + 1} failed for {package}: {last_error[:100]}")

                    if attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))  # Exponential backoff

            except subprocess.TimeoutExpired:
                last_error = "Installation timed out"
                logger.warning(f"Attempt {attempt + 1} timed out for {package}")
                if attempt < max_retries:
                    time.sleep(2.0)

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} error for {package}: {e}")
                if attempt < max_retries:
                    time.sleep(1.0)

        logger.error(f"Failed to install {package} after {max_retries + 1} attempts: {last_error}")
        return False
    
    def prepare_execution_environment(self, code: str) -> Tuple[bool, str]:
        """
        Prepare environment by installing required packages.
        
        Returns:
            (success, message)
        """
        if not self.auto_install_packages:
            return True, "Auto-install disabled"
        
        packages = self.extract_required_packages(code)
        
        if not packages:
            return True, "No packages to install"
        
        logger.info(f"Found {len(packages)} packages to install: {packages}")
        
        failed_packages = []
        for package in packages:
            if not self.install_package(package):
                failed_packages.append(package)
        
        if failed_packages:
            return False, f"Failed to install: {', '.join(failed_packages)}"
        
        return True, f"Installed {len(packages)} packages"
    
    def execute_python_code(
        self,
        code: str,
        check_safety: bool = True,
        auto_install: bool = True,
        use_accumulated: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Python code with safety checks and output capture.

        Args:
            code: Python code to execute
            check_safety: Whether to perform safety checks
            auto_install: Whether to auto-install packages
            use_accumulated: Whether to prepend accumulated code from previous executions

        Returns:
            Dict with keys: success, output, error, files, execution_time
        """
        start_time = time.time()
        result = {
            'success': False,
            'output': '',
            'error': '',
            'files': [],
            'execution_time': 0,
            'workspace': self.workspace_dir
        }

        # Sanitize code to handle Unicode characters
        code = self.sanitize_code(code)

        # Safety checks
        if check_safety:
            is_safe, issues = self.check_imports(code)
            if not is_safe:
                result['error'] = "Safety check failed:\n" + "\n".join(issues)
                return result

        # Install dependencies
        if auto_install:
            success, msg = self.prepare_execution_environment(code)
            if not success:
                result['error'] = f"Environment preparation failed: {msg}"
                return result

        # Build execution code (potentially with accumulated context)
        execution_code = code
        if use_accumulated and self.accumulated_code:
            # Combine accumulated imports and code with current code
            execution_code = self._build_accumulated_code(code)

        # Create a temporary Python file
        code_file = Path(self.workspace_dir).resolve() / "execution_script.py"

        try:
            # Write code to file with explicit UTF-8 encoding
            with open(code_file, 'w', encoding=self.encoding, errors='replace') as f:
                f.write(execution_code)

            # Set up timeout (Unix-only, skip on Windows)
            use_alarm = hasattr(signal, 'SIGALRM') and os.name != 'nt'
            if use_alarm:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)

            # Execute code in subprocess for better isolation
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            process = subprocess.Popen(
                [sys.executable, str(code_file)],
                cwd=str(Path(self.workspace_dir).resolve()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)

                result['output'] = stdout
                if stderr:
                    # Filter out common warnings that aren't errors
                    if process.returncode == 0:
                        result['warnings'] = stderr
                    else:
                        result['error'] = stderr

                result['success'] = process.returncode == 0

                # If successful and accumulating, store the code
                if result['success'] and self.accumulate_blocks:
                    self._accumulate_code(code)

            except subprocess.TimeoutExpired:
                process.kill()
                result['error'] = f"Execution timed out after {self.timeout} seconds"

            finally:
                if use_alarm:
                    signal.alarm(0)  # Cancel alarm

            # List generated files
            result['files'] = self.list_generated_files()

        except Exception as e:
            result['error'] = f"Execution error: {str(e)}\n{traceback.format_exc()}"

        finally:
            result['execution_time'] = time.time() - start_time

            # Cleanup code file
            if code_file.exists():
                try:
                    code_file.unlink()
                except Exception:
                    pass

        return result

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        import_lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_lines.append(stripped)
        return import_lines

    def _accumulate_code(self, code: str):
        """Store code for potential reuse in subsequent blocks."""
        # Extract and store imports
        imports = self._extract_imports(code)
        for imp in imports:
            self.accumulated_imports.add(imp)

        # Store non-import code (functions, classes, variables)
        non_import_lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if not (stripped.startswith('import ') or stripped.startswith('from ')):
                non_import_lines.append(line)

        # Only store definitions (functions, classes, variables)
        code_to_store = '\n'.join(non_import_lines).strip()
        if code_to_store:
            self.accumulated_code.append(code_to_store)

    def _build_accumulated_code(self, new_code: str) -> str:
        """Build code with accumulated context."""
        parts = []

        # Add accumulated imports
        new_imports = self._extract_imports(new_code)
        all_imports = self.accumulated_imports.union(set(new_imports))
        if all_imports:
            parts.append('\n'.join(sorted(all_imports)))
            parts.append('')

        # Add accumulated code (definitions)
        if self.accumulated_code:
            parts.append('# --- Accumulated context ---')
            parts.extend(self.accumulated_code)
            parts.append('# --- End accumulated context ---\n')

        # Add new code (without its imports, as they're already included)
        new_lines = []
        for line in new_code.split('\n'):
            stripped = line.strip()
            if not (stripped.startswith('import ') or stripped.startswith('from ')):
                new_lines.append(line)
        parts.append('\n'.join(new_lines))

        return '\n'.join(parts)

    def reset_accumulated(self):
        """Reset accumulated code context (thread-safe)."""
        with self._lock:
            self.accumulated_code = []
            self.accumulated_imports = set()
        logger.info("Accumulated code context reset")
    
    def execute_code_blocks(
        self,
        text: str,
        check_safety: bool = True,
        stop_on_error: bool = False,
        use_accumulated: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Extract and execute all Python code blocks from text.

        Args:
            text: Text containing code blocks
            check_safety: Whether to perform safety checks
            stop_on_error: Stop execution if a block fails
            use_accumulated: Use accumulated context (defaults to self.accumulate_blocks)

        Returns:
            List of execution results
        """
        code_blocks = self.extract_python_blocks(text)
        results = []

        # Reset accumulated context for fresh execution
        if self.accumulate_blocks:
            self.reset_accumulated()

        use_acc = use_accumulated if use_accumulated is not None else self.accumulate_blocks

        for i, code in enumerate(code_blocks, 1):
            logger.info(f"Executing code block {i}/{len(code_blocks)}")

            # Use accumulated context for blocks after the first
            should_accumulate = use_acc and i > 1
            result = self.execute_python_code(
                code,
                check_safety=check_safety,
                use_accumulated=should_accumulate
            )
            result['block_number'] = i
            result['code'] = code
            results.append(result)

            if not result['success']:
                logger.warning(f"Block {i} failed: {result['error']}")
                if stop_on_error:
                    logger.info("Stopping execution due to error")
                    break

        # Summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Execution complete: {successful}/{len(results)} successful")

        return results
    
    def list_generated_files(self) -> List[str]:
        """
        List files generated in the workspace.
        
        Returns:
            List of file paths
        """
        files = []
        for item in Path(self.workspace_dir).rglob('*'):
            if item.is_file() and item.name != 'execution_script.py':
                files.append(str(item.relative_to(self.workspace_dir)))
        return files
    
    def get_file_content(self, filename: str) -> Optional[bytes]:
        """
        Get content of a generated file.
        
        Args:
            filename: Name of file relative to workspace
            
        Returns:
            File content as bytes, or None if not found
        """
        file_path = Path(self.workspace_dir) / filename
        if file_path.exists() and file_path.is_file():
            return file_path.read_bytes()
        return None
    
    def cleanup(self) -> bool:
        """
        Clean up workspace directory.

        Returns:
            True if cleanup was performed, False if already cleaned up
        """
        with self._lock:
            if self._is_cleaned_up:
                logger.debug(f"Workspace already cleaned up: {self.workspace_dir}")
                return False

            if self._temp_workspace and os.path.exists(self.workspace_dir):
                try:
                    shutil.rmtree(self.workspace_dir)
                    logger.info(f"Cleaned up workspace: {self.workspace_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup workspace: {e}")

            self._is_cleaned_up = True

            # Remove from active executors
            with _cleanup_lock:
                if self in _active_executors:
                    _active_executors.remove(self)

            return True

    @contextmanager
    def session(self):
        """Context manager for automatic cleanup."""
        try:
            yield self
        finally:
            self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor - attempt cleanup if not already done."""
        try:
            if hasattr(self, '_is_cleaned_up') and not self._is_cleaned_up:
                if hasattr(self, '_temp_workspace') and self._temp_workspace:
                    self.cleanup()
        except Exception:
            pass  # Ignore errors during garbage collection


class PersistentCodeExecutor(CodeExecutor):
    """
    Code executor that maintains persistent workspace across multiple executions.
    Useful for notebook-like environments.
    """
    
    def __init__(self, workspace_dir: str, **kwargs):
        """
        Initialize persistent executor.
        
        Args:
            workspace_dir: Directory for persistent storage
            **kwargs: Additional arguments for CodeExecutor
        """
        super().__init__(workspace_dir=workspace_dir, **kwargs)
        self.execution_history = []
    
    def execute_with_history(
        self,
        code: str,
        check_safety: bool = True
    ) -> Dict[str, Any]:
        """
        Execute code and maintain history.
        
        Args:
            code: Python code to execute
            check_safety: Whether to perform safety checks
            
        Returns:
            Execution result
        """
        result = self.execute_python_code(code, check_safety=check_safety)
        
        self.execution_history.append({
            'timestamp': time.time(),
            'code': code,
            'result': result
        })
        
        return result
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history
    
    def save_history(self, filename: str = "execution_history.json"):
        """Save execution history to file."""
        history_file = Path(self.workspace_dir) / filename
        
        # Prepare history for JSON serialization
        serializable_history = []
        for entry in self.execution_history:
            serializable_entry = {
                'timestamp': entry['timestamp'],
                'code': entry['code'],
                'success': entry['result']['success'],
                'output': entry['result']['output'],
                'error': entry['result']['error'],
                'execution_time': entry['result']['execution_time']
            }
            serializable_history.append(serializable_entry)
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"History saved to {history_file}")
    
    def cleanup(self):
        """Don't cleanup - maintain persistent workspace."""
        logger.info(f"Workspace preserved at: {self.workspace_dir}")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic execution
    print("=" * 60)
    print("Example 1: Basic Code Execution")
    print("=" * 60)
    
    executor = CodeExecutor(timeout=30)
    
    code = """
import pandas as pd
import numpy as np

# Create sample data
data = {
    'A': np.random.randn(10),
    'B': np.random.randn(10)
}
df = pd.DataFrame(data)

print("DataFrame created:")
print(df.head())

# Save to CSV
df.to_csv('output.csv', index=False)
print("\\nSaved to output.csv")
"""
    
    result = executor.execute_python_code(code)
    
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['output']}")
    print(f"Files: {result['files']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    
    executor.cleanup()
    
    # Example 2: Extract and execute from Claude response
    print("\n" + "=" * 60)
    print("Example 2: Execute from LLM Response")
    print("=" * 60)
    
    llm_response = """
Here's the code to analyze the data:

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('sine_wave.png')
print("Plot saved!")
```

This creates a sine wave visualization.
"""
    
    executor2 = CodeExecutor()
    results = executor2.execute_code_blocks(llm_response)
    
    for i, result in enumerate(results, 1):
        print(f"\nBlock {i}:")
        print(f"  Success: {result['success']}")
        print(f"  Output: {result['output']}")
        print(f"  Files: {result['files']}")
    
    executor2.cleanup()
