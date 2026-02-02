"""
bedrock_agent_skills - Production-grade skill execution framework for AWS Bedrock Claude.

This module provides a comprehensive framework for executing Claude-generated code
with proper skill management, safety controls, and file handling.

Main Components:
- CodeExecutor: Secure code execution with configurable safety levels
- SkillLoader: Load and manage custom skills with references and assets
- ClaudeCodeAgent: Basic agent for Claude API + code execution
- SkillBasedAgent: Advanced agent with full skill support

Usage:
    from bedrock_agent_skills import SkillBasedAgent, CodeExecutor, SkillLoader

    # Create a skill-based agent
    agent = SkillBasedAgent(
        claude_client=your_claude_client,
        skills_directory="./custom_skills",
        security_level='moderate',  # 'strict', 'moderate', or 'permissive'
        accumulate_blocks=True,      # Execute blocks with accumulated context
    )

    # List available skills
    skills = agent.list_skills()

    # Execute with a skill
    result = agent.execute_with_skill(
        skill_name="analyzing-time-series",
        prompt="Analyze this data",
        file_path="./data/sales.csv",
        include_references=True,
        include_scripts=True
    )

Security Levels:
- 'strict': Blocks subprocess, eval, exec, etc.
- 'moderate': Allows subprocess, blocks eval/exec (recommended for skills)
- 'permissive': Minimal restrictions (use with caution)
"""

__version__ = "1.0.0"
__author__ = "Skills Team"

# Core components
from .code_executor import (
    CodeExecutor,
    PersistentCodeExecutor,
    SECURITY_LEVELS,
    DEFAULT_SAFE_IMPORTS,
    TimeoutException,
)

from .skill_loader import (
    SkillLoader,
    CODE_GENERATION_GUIDELINES,
)

from .claude_code_integration import (
    ClaudeCodeAgent,
    SkillBasedAgent,
)

# Convenience exports
__all__ = [
    # Executors
    "CodeExecutor",
    "PersistentCodeExecutor",

    # Skill management
    "SkillLoader",

    # Agents
    "ClaudeCodeAgent",
    "SkillBasedAgent",

    # Configuration
    "SECURITY_LEVELS",
    "DEFAULT_SAFE_IMPORTS",
    "CODE_GENERATION_GUIDELINES",

    # Exceptions
    "TimeoutException",

    # Version
    "__version__",
]


def get_version():
    """Return the package version."""
    return __version__


def create_agent(
    claude_client,
    skills_directory: str = "./custom_skills",
    security_level: str = 'moderate',
    **kwargs
):
    """
    Factory function to create a SkillBasedAgent with sensible defaults.

    Args:
        claude_client: Your Claude API client (e.g., ClaudeAPI instance)
        skills_directory: Path to custom skills folder
        security_level: 'strict', 'moderate', or 'permissive'
        **kwargs: Additional arguments passed to SkillBasedAgent

    Returns:
        SkillBasedAgent instance
    """
    return SkillBasedAgent(
        claude_client=claude_client,
        skills_directory=skills_directory,
        security_level=security_level,
        auto_execute=True,
        require_confirmation=False,
        persistent=True,
        **kwargs
    )
