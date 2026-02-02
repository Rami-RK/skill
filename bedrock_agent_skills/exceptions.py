"""
Exceptions and validators for bedrock_agent_skills module.

Production-grade exception hierarchy and input validation.
"""

from typing import Any, List, Optional, Union
import os
from pathlib import Path


# =============================================================================
# Custom Exceptions
# =============================================================================

class BedrockSkillsError(Exception):
    """Base exception for all bedrock_agent_skills errors."""
    pass


class ConfigurationError(BedrockSkillsError):
    """Raised when configuration is invalid."""
    pass


class ExecutionError(BedrockSkillsError):
    """Raised when code execution fails."""

    def __init__(self, message: str, code: Optional[str] = None,
                 stderr: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.stderr = stderr


class TimeoutError(ExecutionError):
    """Raised when code execution times out."""
    pass


class SecurityError(BedrockSkillsError):
    """Raised when code fails security checks."""

    def __init__(self, message: str, violations: Optional[List[str]] = None):
        super().__init__(message)
        self.violations = violations or []


class SkillError(BedrockSkillsError):
    """Base exception for skill-related errors."""
    pass


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be found."""
    pass


class SkillValidationError(SkillError):
    """Raised when a skill fails validation."""

    def __init__(self, skill_name: str, issues: List[str]):
        message = f"Skill '{skill_name}' validation failed: {', '.join(issues)}"
        super().__init__(message)
        self.skill_name = skill_name
        self.issues = issues


class PackageInstallError(ExecutionError):
    """Raised when package installation fails."""

    def __init__(self, package: str, reason: str):
        super().__init__(f"Failed to install package '{package}': {reason}")
        self.package = package
        self.reason = reason


# =============================================================================
# Validators
# =============================================================================

VALID_SECURITY_LEVELS = {'strict', 'moderate', 'permissive'}
VALID_ENCODINGS = {'utf-8', 'utf-16', 'ascii', 'latin-1', 'cp1252'}


def validate_security_level(level: str) -> str:
    """
    Validate security level parameter.

    Args:
        level: Security level string

    Returns:
        Validated security level

    Raises:
        ConfigurationError: If level is invalid
    """
    if not isinstance(level, str):
        raise ConfigurationError(
            f"security_level must be a string, got {type(level).__name__}"
        )

    level = level.lower().strip()
    if level not in VALID_SECURITY_LEVELS:
        raise ConfigurationError(
            f"Invalid security_level '{level}'. "
            f"Must be one of: {', '.join(sorted(VALID_SECURITY_LEVELS))}"
        )

    return level


def validate_timeout(timeout: Union[int, float]) -> int:
    """
    Validate timeout parameter.

    Args:
        timeout: Timeout value in seconds

    Returns:
        Validated timeout as integer

    Raises:
        ConfigurationError: If timeout is invalid
    """
    if not isinstance(timeout, (int, float)):
        raise ConfigurationError(
            f"timeout must be a number, got {type(timeout).__name__}"
        )

    timeout = int(timeout)
    if timeout <= 0:
        raise ConfigurationError(
            f"timeout must be positive, got {timeout}"
        )

    if timeout > 3600:  # 1 hour max
        raise ConfigurationError(
            f"timeout exceeds maximum of 3600 seconds, got {timeout}"
        )

    return timeout


def validate_encoding(encoding: str) -> str:
    """
    Validate encoding parameter.

    Args:
        encoding: Encoding string

    Returns:
        Validated encoding

    Raises:
        ConfigurationError: If encoding is invalid
    """
    if not isinstance(encoding, str):
        raise ConfigurationError(
            f"encoding must be a string, got {type(encoding).__name__}"
        )

    encoding = encoding.lower().strip()

    # Try to validate by attempting to use it
    try:
        "test".encode(encoding)
    except LookupError:
        raise ConfigurationError(
            f"Invalid encoding '{encoding}'. "
            f"Common encodings: {', '.join(sorted(VALID_ENCODINGS))}"
        )

    return encoding


def validate_workspace_dir(workspace_dir: Optional[str]) -> Optional[str]:
    """
    Validate workspace directory parameter.

    Args:
        workspace_dir: Workspace directory path

    Returns:
        Validated absolute path or None

    Raises:
        ConfigurationError: If path is invalid
    """
    if workspace_dir is None:
        return None

    if not isinstance(workspace_dir, (str, Path)):
        raise ConfigurationError(
            f"workspace_dir must be a string or Path, got {type(workspace_dir).__name__}"
        )

    path = Path(workspace_dir)

    # Check for obviously problematic paths
    resolved = path.resolve()
    str_path = str(resolved)

    # Prevent writing to system directories
    dangerous_prefixes = ['/bin', '/sbin', '/usr/bin', '/usr/sbin', '/etc',
                          '/boot', '/dev', '/proc', '/sys']

    if os.name != 'nt':  # Unix-like systems
        for prefix in dangerous_prefixes:
            if str_path.startswith(prefix):
                raise ConfigurationError(
                    f"workspace_dir cannot be in system directory: {prefix}"
                )

    return str(resolved)


def validate_skills_directory(skills_dir: str) -> Path:
    """
    Validate skills directory parameter.

    Args:
        skills_dir: Skills directory path

    Returns:
        Validated Path object

    Raises:
        ConfigurationError: If directory doesn't exist or is invalid
    """
    if not isinstance(skills_dir, (str, Path)):
        raise ConfigurationError(
            f"skills_directory must be a string or Path, got {type(skills_dir).__name__}"
        )

    path = Path(skills_dir)

    if not path.exists():
        raise ConfigurationError(
            f"Skills directory does not exist: {path}"
        )

    if not path.is_dir():
        raise ConfigurationError(
            f"Skills path is not a directory: {path}"
        )

    return path


def validate_max_memory(max_memory_mb: int) -> int:
    """
    Validate maximum memory parameter.

    Args:
        max_memory_mb: Maximum memory in MB

    Returns:
        Validated memory limit

    Raises:
        ConfigurationError: If value is invalid
    """
    if not isinstance(max_memory_mb, int):
        raise ConfigurationError(
            f"max_memory_mb must be an integer, got {type(max_memory_mb).__name__}"
        )

    if max_memory_mb <= 0:
        raise ConfigurationError(
            f"max_memory_mb must be positive, got {max_memory_mb}"
        )

    if max_memory_mb > 65536:  # 64GB max
        raise ConfigurationError(
            f"max_memory_mb exceeds maximum of 65536 MB, got {max_memory_mb}"
        )

    return max_memory_mb


# =============================================================================
# Retry Decorator
# =============================================================================

import time
import functools
import logging
from typing import Callable, TypeVar, Type

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator for retrying a function on exception.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch
        logger: Logger for retry messages

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if logger:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if logger:
                            logger.error(
                                f"All {max_retries + 1} attempts failed for {func.__name__}"
                            )

            raise last_exception

        return wrapper
    return decorator
