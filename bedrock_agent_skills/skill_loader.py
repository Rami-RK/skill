"""
Enhanced Skill Loader for AWS Bedrock
Properly loads skill instructions, reference files, and assets.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SkillLoader:
    """
    Loads skill instructions and all associated assets.
    Replicates the behavior of Anthropic's Skills API for Bedrock.
    """
    
    def __init__(self, skills_directory: str = "./custom_skills"):
        """
        Initialize skill loader.
        
        Args:
            skills_directory: Base directory containing skill folders
        """
        self.skills_directory = Path(skills_directory)
        self.skill_cache = {}
    
    def load_skill(self, skill_name: str, include_scripts: bool = False) -> Dict[str, any]:
        """
        Load a complete skill with all its assets.
        
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
        """
        if skill_name in self.skill_cache:
            logger.info(f"Using cached skill: {skill_name}")
            return self.skill_cache[skill_name]
        
        skill_path = self.skills_directory / skill_name
        
        if not skill_path.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_path}")
        
        skill_data = {
            'name': skill_name,
            'path': str(skill_path),
            'instructions': None,
            'references': {},
            'scripts': {},
            'assets': [],
            'metadata': {}
        }
        
        # Load main SKILL.md
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_path}")
        
        with open(skill_md, 'r', encoding='utf-8') as f:
            content = f.read()
            skill_data['instructions'] = content
            skill_data['metadata'] = self._extract_metadata(content)
        
        # Load reference files (.md files other than SKILL.md)
        for md_file in skill_path.glob("*.md"):
            if md_file.name != "SKILL.md":
                with open(md_file, 'r', encoding='utf-8') as f:
                    skill_data['references'][md_file.name] = f.read()
                logger.info(f"Loaded reference: {md_file.name}")
        
        # Load script files if requested
        scripts_dir = skill_path / "scripts"
        if include_scripts and scripts_dir.exists():
            for script_file in scripts_dir.rglob("*.py"):
                if script_file.name != "__init__.py":
                    with open(script_file, 'r', encoding='utf-8') as f:
                        rel_path = script_file.relative_to(scripts_dir)
                        skill_data['scripts'][str(rel_path)] = f.read()
                    logger.info(f"Loaded script: {rel_path}")
        
        # List other asset files
        for asset_file in skill_path.rglob("*"):
            if asset_file.is_file() and asset_file.suffix not in ['.md', '.py', '.txt']:
                rel_path = asset_file.relative_to(skill_path)
                skill_data['assets'].append(str(rel_path))
        
        # Cache the skill
        self.skill_cache[skill_name] = skill_data
        
        logger.info(f"Loaded skill '{skill_name}': {len(skill_data['references'])} refs, "
                   f"{len(skill_data['scripts'])} scripts, {len(skill_data['assets'])} assets")
        
        return skill_data
    
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
        include_scripts: bool = False
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
            
        Returns:
            Complete prompt string
        """
        skill_data = self.load_skill(skill_name, include_scripts=include_scripts)
        
        prompt_parts = []
        
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
                prompt_parts.append("Available helper scripts (content included):\n")
                for script_path, script_content in skill_data['scripts'].items():
                    prompt_parts.append(f"\n## Script: scripts/{script_path}\n")
                    prompt_parts.append(f"```python\n{script_content}\n```\n")
            else:
                prompt_parts.append("Available helper scripts in the skill directory:\n")
                for script_path in skill_data['scripts'].keys():
                    prompt_parts.append(f"- scripts/{script_path}")
            prompt_parts.append("</skill_scripts>\n")
        
        # Add asset information
        if skill_data['assets']:
            prompt_parts.append("<skill_assets>")
            prompt_parts.append("Additional assets available in this skill:\n")
            for asset in skill_data['assets']:
                prompt_parts.append(f"- {asset}")
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
    
    def get_skill_info(self, skill_name: str) -> Dict[str, any]:
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
            'assets': []
        }
        
        # List reference files
        for md_file in skill_path.glob("*.md"):
            if md_file.name != "SKILL.md":
                info['references'].append(md_file.name)
        
        # List assets
        for asset in skill_path.rglob("*"):
            if asset.is_file() and asset.suffix not in ['.md', '.py', '.txt']:
                info['assets'].append(str(asset.relative_to(skill_path)))
        
        # Get metadata if SKILL.md exists
        if info['has_skill_md']:
            with open(skill_path / "SKILL.md", 'r', encoding='utf-8') as f:
                content = f.read()
                info['metadata'] = self._extract_metadata(content)
        
        return info
    
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
