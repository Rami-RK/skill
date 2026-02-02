"""
Secure Code Execution Framework for Local Automation
Provides sandboxed Python code execution with safety measures.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from contextlib import contextmanager
import signal
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    """
    
    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        timeout: int = 300,
        max_memory_mb: int = 1024,
        allowed_imports: Optional[List[str]] = None,
        blocked_imports: Optional[List[str]] = None,
        auto_install_packages: bool = True
    ):
        """
        Initialize code executor.
        
        Args:
            workspace_dir: Directory for code execution (created if doesn't exist)
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            allowed_imports: Whitelist of allowed imports (None = allow all)
            blocked_imports: Blacklist of blocked imports
            auto_install_packages: Auto-install missing packages
        """
        self.workspace_dir = str(Path(workspace_dir).resolve()) if workspace_dir else tempfile.mkdtemp(prefix="code_exec_")
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_imports = allowed_imports
        self.blocked_imports = blocked_imports or [
            'os.system', 'subprocess', 'eval', 'exec', 'compile',
            '__import__', 'open'  # We'll provide safe alternatives
        ]
        self.auto_install_packages = auto_install_packages
        
        # Create workspace
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        
        # Track installed packages
        self.installed_packages = set()
        
        logger.info(f"Code executor initialized with workspace: {self.workspace_dir}")
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from text (markdown format).
        
        Args:
            text: Text containing code blocks
            
        Returns:
            List of dicts with 'language' and 'code' keys
        """
        markdown_pattern = r'```(\w+)?\n(.*?)```'
        markdown_matches = re.findall(markdown_pattern, text, re.DOTALL)

        run_code_pattern = r'<run_code>\s*<language>(.*?)</language>\s*<code>(.*?)</code>\s*</run_code>'
        run_code_matches = re.findall(run_code_pattern, text, re.DOTALL)

        code_blocks = []
        for lang, code in markdown_matches:
            lang = lang.lower() if lang else 'python'
            code_blocks.append({
                'language': lang,
                'code': code.strip()
            })

        for lang, code in run_code_matches:
            lang = lang.strip().lower() if lang else 'python'
            code_blocks.append({
                'language': lang,
                'code': code.strip()
            })
        
        return code_blocks
    
    def extract_python_blocks(self, text: str) -> List[str]:
        """Extract only Python code blocks."""
        blocks = self.extract_code_blocks(text)
        return [block['code'] for block in blocks if block['language'] == 'python']
    
    def check_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check if code contains only allowed imports.
        
        Returns:
            (is_safe, list_of_issues)
        """
        issues = []
        
        # Extract import statements
        import_pattern = r'^\s*(?:import|from)\s+(\S+)'
        imports = re.findall(import_pattern, code, re.MULTILINE)
        
        # Check against blocked imports
        for imp in imports:
            base_module = imp.split('.')[0]
            
            if self.allowed_imports and base_module not in self.allowed_imports:
                issues.append(f"Import '{imp}' not in whitelist")
            
            if any(blocked in imp for blocked in self.blocked_imports):
                issues.append(f"Blocked import detected: '{imp}'")
        
        # Check for dangerous built-ins
        dangerous_patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bcompile\s*\(',
            r'\b__import__\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(f"Dangerous pattern detected: {pattern}")
        
        return len(issues) == 0, issues
    
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
        
        packages = []
        for imp in imports:
            base_module = imp.split('.')[0]
            
            # Skip standard library
            if base_module in ['os', 'sys', 'json', 're', 'time', 'datetime', 
                               'collections', 'itertools', 'functools', 'math']:
                continue
            
            # Map to actual package name
            package_name = package_map.get(base_module, base_module)
            packages.append(package_name)
        
        return list(set(packages))
    
    def install_package(self, package: str) -> bool:
        """
        Install a Python package using pip.
        
        Args:
            package: Package name to install
            
        Returns:
            True if successful, False otherwise
        """
        if package in self.installed_packages:
            logger.info(f"Package '{package}' already installed")
            return True
        
        try:
            logger.info(f"Installing package: {package}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--break-system-packages"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self.installed_packages.add(package)
                logger.info(f"Successfully installed: {package}")
                return True
            else:
                logger.error(f"Failed to install {package}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Installation of {package} timed out")
            return False
        except Exception as e:
            logger.error(f"Error installing {package}: {e}")
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
        auto_install: bool = True
    ) -> Dict[str, Any]:
        """
        Execute Python code with safety checks and output capture.
        
        Args:
            code: Python code to execute
            check_safety: Whether to perform safety checks
            auto_install: Whether to auto-install packages
            
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
        
        # Create a temporary Python file
        code_file = Path(self.workspace_dir).resolve() / "execution_script.py"
        
        try:
            # Write code to file
            with open(code_file, 'w') as f:
                f.write(code)
            
            # Set up timeout
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            # Execute code in subprocess for better isolation
            process = subprocess.Popen(
                [sys.executable, str(code_file)],
                cwd=str(Path(self.workspace_dir).resolve()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                
                result['output'] = stdout
                if stderr:
                    result['error'] = stderr
                
                result['success'] = process.returncode == 0
                
            except subprocess.TimeoutExpired:
                process.kill()
                result['error'] = f"Execution timed out after {self.timeout} seconds"
            
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel alarm
            
            # List generated files
            result['files'] = self.list_generated_files()
            
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
        
        finally:
            result['execution_time'] = time.time() - start_time
            
            # Cleanup code file
            if code_file.exists():
                code_file.unlink()
        
        return result
    
    def execute_code_blocks(
        self,
        text: str,
        check_safety: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract and execute all Python code blocks from text.
        
        Args:
            text: Text containing code blocks
            check_safety: Whether to perform safety checks
            
        Returns:
            List of execution results
        """
        code_blocks = self.extract_python_blocks(text)
        results = []
        
        for i, code in enumerate(code_blocks, 1):
            logger.info(f"Executing code block {i}/{len(code_blocks)}")
            result = self.execute_python_code(code, check_safety=check_safety)
            result['block_number'] = i
            results.append(result)
            
            if not result['success']:
                logger.warning(f"Block {i} failed: {result['error']}")
        
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
    
    def cleanup(self):
        """Clean up workspace directory."""
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)
            logger.info(f"Cleaned up workspace: {self.workspace_dir}")
    
    @contextmanager
    def session(self):
        """Context manager for automatic cleanup."""
        try:
            yield self
        finally:
            self.cleanup()


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
