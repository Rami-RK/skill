"""
Enhanced Skill Loader for AWS Bedrock
Properly loads skill instructions, reference files, and assets.

Production-grade module for AWS Bedrock Claude skill execution.
Supports multiple skill directory structures for flexibility.
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import hashlib

from .exceptions import (
    ConfigurationError,
    SkillError,
    SkillNotFoundError,
    SkillValidationError,
    validate_skills_directory,
)

# Use module-level logger (don't configure basicConfig - leave to caller)
logger = logging.getLogger(__name__)

# Default prompt additions for better code generation
CODE_GENERATION_GUIDELINES = """
<code_generation_guidelines>
When generating Python code, follow these critical rules:

1. **Self-Contained Code Blocks**: Each code block MUST be fully self-contained and executable independently.
   - Include ALL required imports at the top of EVERY code block
   - Do NOT assume variables from previous code blocks exist
   - Define all necessary functions and classes within each block

2. **Output Files**: Save all generated files to the current working directory or a specified output path.
   - Use relative paths for output files
   - Create directories if needed using os.makedirs(path, exist_ok=True)

3. **Avoid Unicode Issues**: Do not use emojis or special Unicode characters in code.
   - Use ASCII-only characters in strings and comments
   - Replace checkmarks with [OK], X marks with [X], etc.

4. **Error Handling**: Include basic error handling for file operations and external dependencies.

5. **Dependencies**: Only use standard library or commonly available packages (numpy, pandas, matplotlib, etc.)
</code_generation_guidelines>
"""


class SkillLoader:
    """
    Loads skill instructions and all associated assets.
    Replicates the behavior of Anthropic's Skills API for Bedrock.

    Thread-safe implementation with caching support.
    """

    def __init__(self, skills_directory: str = "./custom_skills"):
        """
        Initialize skill loader.

        Args:
            skills_directory: Base directory containing skill folders

        Raises:
            ConfigurationError: If skills_directory is invalid or doesn't exist
        """
        # Validate but don't require it to exist (may be created later)
        if not isinstance(skills_directory, (str, Path)):
            raise ConfigurationError(
                f"skills_directory must be a string or Path, got {type(skills_directory).__name__}"
            )

        self.skills_directory = Path(skills_directory)

        # Thread-safe cache
        self._lock = threading.RLock()
        self.skill_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_skill(self, skill_name: str, include_scripts: bool = False) -> Dict[str, Any]:
        """
        Load a complete skill with all its assets.

        Supports two directory structures:
        1. Flat structure: references directly in skill folder
        2. Nested structure: references in references/ subfolder (L5 style)

        Args:
            skill_name: Name of the skill directory
            include_scripts: Whether to include script file contents

        Returns:
            Dict containing:
                - instructions: Main SKILL.md content
                - references: Dict of reference file contents
                - scripts: Dict of script file contents (if include_scripts=True)
                - assets: List of asset file paths
                - metadata: Skill metadata

        Raises:
            SkillNotFoundError: If skill directory or SKILL.md not found
            SkillError: If skill loading fails
        """
        if not skill_name or not isinstance(skill_name, str):
            raise SkillError("skill_name must be a non-empty string")

        # Sanitize skill name (prevent path traversal)
        skill_name = skill_name.strip().replace('..', '').replace('/', '').replace('\\', '')
        if not skill_name:
            raise SkillError("Invalid skill_name after sanitization")

        cache_key = f"{skill_name}_{include_scripts}"

        # Thread-safe cache check
        with self._lock:
            if cache_key in self.skill_cache:
                logger.debug(f"Using cached skill: {skill_name}")
                return self.skill_cache[cache_key]

        skill_path = self.skills_directory / skill_name

        if not skill_path.exists():
            raise SkillNotFoundError(f"Skill directory not found: {skill_path}")

        skill_data = {
            'name': skill_name,
            'path': str(skill_path),
            'instructions': None,
            'references': {},
            'scripts': {},
            'assets': [],
            'asset_contents': {},
            'metadata': {}
        }

        # Load main SKILL.md
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            raise SkillNotFoundError(f"SKILL.md not found in {skill_path}")

        with open(skill_md, 'r', encoding='utf-8') as f:
            content = f.read()
            skill_data['instructions'] = content
            skill_data['metadata'] = self._extract_metadata(content)

        # Load reference files - check both flat and nested structures
        self._load_references(skill_path, skill_data)

        # Load script files if requested
        scripts_dir = skill_path / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.rglob("*.py"):
                if script_file.name != "__init__.py":
                    rel_path = script_file.relative_to(scripts_dir)
                    if include_scripts:
                        with open(script_file, 'r', encoding='utf-8') as f:
                            skill_data['scripts'][str(rel_path)] = f.read()
                        logger.info(f"Loaded script content: {rel_path}")
                    else:
                        # Just record the path
                        skill_data['scripts'][str(rel_path)] = None

        # Load asset files
        self._load_assets(skill_path, skill_data)

        # Thread-safe cache update
        with self._lock:
            self.skill_cache[cache_key] = skill_data

        logger.info(f"Loaded skill '{skill_name}': {len(skill_data['references'])} refs, "
                   f"{len([s for s in skill_data['scripts'].values() if s is not None])} scripts loaded, "
                   f"{len(skill_data['assets'])} assets")

        return skill_data

    def clear_cache(self, skill_name: Optional[str] = None):
        """
        Clear skill cache.

        Args:
            skill_name: Specific skill to clear, or None to clear all
        """
        with self._lock:
            if skill_name:
                keys_to_remove = [k for k in self.skill_cache if k.startswith(f"{skill_name}_")]
                for key in keys_to_remove:
                    del self.skill_cache[key]
                logger.debug(f"Cleared cache for skill: {skill_name}")
            else:
                self.skill_cache.clear()
                logger.debug("Cleared all skill cache")

    def _load_references(self, skill_path: Path, skill_data: Dict) -> None:
        """Load reference files from both flat and nested structures."""
        # First, check for references/ subdirectory (L5 style)
        refs_dir = skill_path / "references"
        if refs_dir.exists():
            for md_file in refs_dir.rglob("*.md"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    skill_data['references'][md_file.name] = f.read()
                logger.info(f"Loaded reference (nested): {md_file.name}")

        # Also check for .md files directly in skill folder (flat style)
        for md_file in skill_path.glob("*.md"):
            if md_file.name != "SKILL.md" and md_file.name not in skill_data['references']:
                with open(md_file, 'r', encoding='utf-8') as f:
                    skill_data['references'][md_file.name] = f.read()
                logger.info(f"Loaded reference (flat): {md_file.name}")

    def _load_assets(self, skill_path: Path, skill_data: Dict) -> None:
        """Load asset files and their contents."""
        # Check assets/ subdirectory
        assets_dir = skill_path / "assets"
        if assets_dir.exists():
            for asset_file in assets_dir.rglob("*"):
                if asset_file.is_file():
                    rel_path = str(asset_file.relative_to(skill_path))
                    skill_data['assets'].append(rel_path)

                    # Load text assets (templates, etc.)
                    if asset_file.suffix in ['.md', '.txt', '.tex', '.json', '.yaml', '.yml']:
                        try:
                            with open(asset_file, 'r', encoding='utf-8') as f:
                                skill_data['asset_contents'][rel_path] = f.read()
                            logger.info(f"Loaded asset content: {rel_path}")
                        except UnicodeDecodeError:
                            logger.warning(f"Could not read asset as text: {rel_path}")

        # Also check for other file types in skill directory
        for item in skill_path.rglob("*"):
            if item.is_file():
                rel_path = str(item.relative_to(skill_path))
                # Skip already processed files
                if (rel_path.startswith("references/") or
                    rel_path.startswith("scripts/") or
                    item.suffix in ['.md', '.py'] or
                    rel_path in skill_data['assets']):
                    continue

                if item.suffix not in ['.pyc', '.pyo', '__pycache__']:
                    skill_data['assets'].append(rel_path)
    
    def _extract_metadata(self, skill_content: str) -> Dict[str, str]:
        """
        Extract metadata from SKILL.md front matter.
        
        Args:
            skill_content: Content of SKILL.md
            
        Returns:
            Dict of metadata key-value pairs
        """
        metadata = {}
        
        if skill_content.startswith('---'):
            lines = skill_content.split('\n')
            in_frontmatter = False
            
            for i, line in enumerate(lines):
                if i == 0 and line.strip() == '---':
                    in_frontmatter = True
                    continue
                
                if in_frontmatter and line.strip() == '---':
                    break
                
                if in_frontmatter and ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip().strip('"')
        
        return metadata
    
    def build_comprehensive_prompt(
        self,
        skill_name: str,
        user_query: str,
        file_content: Optional[str] = None,
        additional_context: Optional[str] = None,
        include_references: bool = True,
        include_scripts: bool = False,
        include_assets: bool = True,
        include_code_guidelines: bool = True,
        workspace_dir: Optional[str] = None
    ) -> str:
        """
        Build a comprehensive prompt with skill instructions and all assets.

        Args:
            skill_name: Name of skill to use
            user_query: User's request
            file_content: Optional input file content
            additional_context: Optional additional context
            include_references: Include reference files in prompt
            include_scripts: Include script contents in prompt
            include_assets: Include asset file contents in prompt
            include_code_guidelines: Include code generation guidelines
            workspace_dir: Working directory for code execution

        Returns:
            Complete prompt string
        """
        skill_data = self.load_skill(skill_name, include_scripts=include_scripts)

        prompt_parts = []

        # Add code generation guidelines first
        if include_code_guidelines:
            prompt_parts.append(CODE_GENERATION_GUIDELINES)

        # Add execution context
        if workspace_dir:
            prompt_parts.append(f"<execution_context>")
            prompt_parts.append(f"Working directory: {workspace_dir}")
            prompt_parts.append(f"Output files should be saved to the current working directory.")
            prompt_parts.append(f"</execution_context>\n")

        # Add main skill instructions
        prompt_parts.append("<skill_instructions>")
        prompt_parts.append(f"# Skill: {skill_data['name']}")
        if skill_data['metadata'].get('description'):
            prompt_parts.append(f"\nDescription: {skill_data['metadata']['description']}\n")
        prompt_parts.append(skill_data['instructions'])
        prompt_parts.append("</skill_instructions>\n")

        # Add reference files
        if include_references and skill_data['references']:
            prompt_parts.append("<skill_reference_files>")
            prompt_parts.append("The following reference files are part of this skill:\n")

            for ref_name, ref_content in skill_data['references'].items():
                prompt_parts.append(f"\n## Reference File: {ref_name}\n")
                prompt_parts.append(ref_content)

            prompt_parts.append("</skill_reference_files>\n")

        # Add script references (content or just paths)
        if skill_data['scripts']:
            prompt_parts.append("<skill_scripts>")
            if include_scripts:
                prompt_parts.append("Available helper scripts (you can use these patterns in your code):\n")
                for script_path, script_content in skill_data['scripts'].items():
                    if script_content:  # Only if content was loaded
                        prompt_parts.append(f"\n## Script: scripts/{script_path}\n")
                        prompt_parts.append(f"```python\n{script_content}\n```\n")
            else:
                prompt_parts.append("Available helper scripts in the skill directory:\n")
                for script_path in skill_data['scripts'].keys():
                    prompt_parts.append(f"- scripts/{script_path}")
            prompt_parts.append("</skill_scripts>\n")

        # Add asset information and contents
        if include_assets and (skill_data['assets'] or skill_data.get('asset_contents')):
            prompt_parts.append("<skill_assets>")
            prompt_parts.append("Assets available in this skill:\n")

            for asset in skill_data['assets']:
                prompt_parts.append(f"- {asset}")

            # Include text asset contents
            if skill_data.get('asset_contents'):
                prompt_parts.append("\n### Asset Contents:\n")
                for asset_path, content in skill_data['asset_contents'].items():
                    prompt_parts.append(f"\n#### {asset_path}\n")
                    prompt_parts.append(f"```\n{content}\n```\n")

            prompt_parts.append("</skill_assets>\n")

        # Add additional context
        if additional_context:
            prompt_parts.append("<additional_context>")
            prompt_parts.append(additional_context)
            prompt_parts.append("</additional_context>\n")

        # Add input file content
        if file_content:
            prompt_parts.append("<input_data>")
            prompt_parts.append(file_content)
            prompt_parts.append("</input_data>\n")

        # Add user query
        prompt_parts.append("<user_request>")
        prompt_parts.append(user_query)
        prompt_parts.append("</user_request>")

        return "\n".join(prompt_parts)
    
    def get_skill_info(self, skill_name: str) -> Dict[str, Any]:
        """
        Get information about a skill without loading all content.

        Args:
            skill_name: Name of skill

        Returns:
            Dict with skill info
        """
        skill_path = self.skills_directory / skill_name

        if not skill_path.exists():
            return None

        info = {
            'name': skill_name,
            'path': str(skill_path),
            'has_skill_md': (skill_path / "SKILL.md").exists(),
            'references': [],
            'has_scripts': (skill_path / "scripts").exists(),
            'scripts': [],
            'assets': [],
            'metadata': {}
        }

        # List reference files (both flat and nested structures)
        refs_dir = skill_path / "references"
        if refs_dir.exists():
            for md_file in refs_dir.rglob("*.md"):
                info['references'].append(md_file.name)

        for md_file in skill_path.glob("*.md"):
            if md_file.name != "SKILL.md" and md_file.name not in info['references']:
                info['references'].append(md_file.name)

        # List scripts
        scripts_dir = skill_path / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.rglob("*.py"):
                if script_file.name != "__init__.py":
                    info['scripts'].append(str(script_file.relative_to(scripts_dir)))

        # List assets
        assets_dir = skill_path / "assets"
        if assets_dir.exists():
            for asset in assets_dir.rglob("*"):
                if asset.is_file():
                    info['assets'].append(str(asset.relative_to(skill_path)))

        # Get metadata if SKILL.md exists
        if info['has_skill_md']:
            with open(skill_path / "SKILL.md", 'r', encoding='utf-8') as f:
                content = f.read()
                info['metadata'] = self._extract_metadata(content)

        return info

    def validate_skill(self, skill_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a skill directory structure.

        Args:
            skill_name: Name of skill to validate

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        skill_path = self.skills_directory / skill_name

        if not skill_path.exists():
            return False, [f"Skill directory not found: {skill_path}"]

        # Check for SKILL.md
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            issues.append("Missing required SKILL.md file")
        else:
            with open(skill_md, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    issues.append("SKILL.md is empty")
                metadata = self._extract_metadata(content)
                if not metadata.get('name'):
                    issues.append("SKILL.md missing 'name' in frontmatter")
                if not metadata.get('description'):
                    issues.append("SKILL.md missing 'description' in frontmatter")

        # Check scripts directory if it exists
        scripts_dir = skill_path / "scripts"
        if scripts_dir.exists():
            init_file = scripts_dir / "__init__.py"
            if not init_file.exists():
                issues.append("scripts/ directory missing __init__.py")

        return len(issues) == 0, issues
    
    def list_skills(self) -> List[str]:
        """List all available skills."""
        if not self.skills_directory.exists():
            return []
        
        skills = []
        for item in self.skills_directory.iterdir():
            if item.is_dir() and (item / "SKILL.md").exists():
                skills.append(item.name)
        
        return sorted(skills)
    
    def get_skill_asset(self, skill_name: str, asset_path: str) -> Optional[bytes]:
        """
        Get the content of a specific asset file.
        
        Args:
            skill_name: Name of skill
            asset_path: Relative path to asset within skill directory
            
        Returns:
            Asset content as bytes, or None if not found
        """
        skill_path = self.skills_directory / skill_name
        asset_file = skill_path / asset_path
        
        if asset_file.exists() and asset_file.is_file():
            return asset_file.read_bytes()
        
        return None


# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    loader = SkillLoader("./custom_skills")
    
    print("=" * 60)
    print("Available Skills")
    print("=" * 60)
    
    skills = loader.list_skills()
    for skill in skills:
        info = loader.get_skill_info(skill)
        print(f"\n{skill}:")
        print(f"  References: {len(info['references'])} files")
        print(f"  Scripts: {'Yes' if info['has_scripts'] else 'No'}")
        print(f"  Assets: {len(info['assets'])} files")
        if info.get('metadata', {}).get('description'):
            print(f"  Description: {info['metadata']['description'][:80]}...")
    
    # Load a skill with all references
    if skills:
        print("\n" + "=" * 60)
        print(f"Loading Skill: {skills[0]}")
        print("=" * 60)
        
        skill_data = loader.load_skill(skills[0])
        
        print(f"Instructions length: {len(skill_data['instructions'])} chars")
        print(f"References loaded: {list(skill_data['references'].keys())}")
        print(f"Scripts available: {len(skill_data['scripts'])}")
        print(f"Assets: {skill_data['assets']}")
        
        # Build comprehensive prompt
        prompt = loader.build_comprehensive_prompt(
            skill_name=skills[0],
            user_query="Test query",
            include_references=True,
            include_scripts=False
        )
        
        print(f"\nTotal prompt length: {len(prompt)} characters")
        print("\nPrompt structure:")
        for tag in ['<skill_instructions>', '<skill_reference_files>', 
                    '<skill_scripts>', '<user_request>']:
            if tag in prompt:
                print(f"  ✓ {tag}")
