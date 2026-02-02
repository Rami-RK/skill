"""
Integration module connecting Claude API with Code Executor.
Provides automated workflow for LLM-generated code execution.

Production-grade module for AWS Bedrock Claude skill execution.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from .code_executor import CodeExecutor, PersistentCodeExecutor, SECURITY_LEVELS
from .skill_loader import SkillLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClaudeCodeAgent:
    """
    Agent that combines Claude API with code execution capabilities.
    Automatically extracts and executes code from Claude's responses.

    Production-grade agent for AWS Bedrock Claude skill workflows.
    """

    def __init__(
        self,
        claude_client,
        workspace_dir: Optional[str] = None,
        auto_execute: bool = True,
        require_confirmation: bool = True,
        persistent: bool = False,
        security_level: str = 'moderate',
        accumulate_blocks: bool = True,
        encoding: str = 'utf-8',
        timeout: int = 300,
        **executor_kwargs
    ):
        """
        Initialize Claude Code Agent.

        Args:
            claude_client: Instance of ClaudeAPI or ClaudeAPIS35
            workspace_dir: Directory for code execution
            auto_execute: Automatically execute code from responses
            require_confirmation: Ask for confirmation before execution
            persistent: Use persistent workspace
            security_level: 'strict', 'moderate', or 'permissive' (default: 'moderate')
            accumulate_blocks: Execute blocks with accumulated context (default: True)
            encoding: File encoding (default: 'utf-8')
            timeout: Execution timeout in seconds (default: 300)
            **executor_kwargs: Additional arguments for CodeExecutor
        """
        self.claude_client = claude_client
        self.auto_execute = auto_execute
        self.require_confirmation = require_confirmation
        self.security_level = security_level

        # Prepare executor kwargs
        exec_kwargs = {
            'security_level': security_level,
            'accumulate_blocks': accumulate_blocks,
            'encoding': encoding,
            'timeout': timeout,
            **executor_kwargs
        }

        # Initialize executor
        if persistent:
            self.executor = PersistentCodeExecutor(
                workspace_dir=workspace_dir or "./claude_workspace",
                **exec_kwargs
            )
        else:
            self.executor = CodeExecutor(
                workspace_dir=workspace_dir,
                **exec_kwargs
            )

        self.conversation_history = []
        logger.info(f"ClaudeCodeAgent initialized with security_level={security_level}")
    
    def ask_and_execute(
        self,
        prompt: str,
        skill_instructions: Optional[str] = None,
        file_content: Optional[str] = None,
        execute: Optional[bool] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send prompt to Claude and optionally execute code in response.
        
        Args:
            prompt: User prompt
            skill_instructions: Optional skill instructions to include
            file_content: Optional file content to include
            execute: Override auto_execute setting
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with response, execution results, and files
        """
        # Build comprehensive prompt
        full_prompt = self._build_prompt(prompt, skill_instructions, file_content)
        
        # Get Claude's response
        logger.info("Sending request to Claude...")
        try:
            response = self.claude_client.invoke_llm_model(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'executions': []
            }
        
        logger.info(f"Received response ({len(response)} chars)")
        
        # Store in history
        self.conversation_history.append({
            'prompt': prompt,
            'response': response
        })
        
        # Extract code blocks
        code_blocks = self.executor.extract_python_blocks(response)
        
        result = {
            'success': True,
            'response': response,
            'code_blocks': code_blocks,
            'executions': [],
            'files': []
        }
        
        # Execute code if requested
        should_execute = execute if execute is not None else self.auto_execute
        
        if should_execute and code_blocks:
            logger.info(f"Found {len(code_blocks)} code blocks")
            
            if self.require_confirmation:
                print("\n" + "=" * 60)
                print("CODE FOUND IN RESPONSE")
                print("=" * 60)
                for i, code in enumerate(code_blocks, 1):
                    print(f"\nBlock {i}:")
                    print("-" * 40)
                    print(code[:200] + ("..." if len(code) > 200 else ""))
                print("\n" + "=" * 60)
                
                response = input("Execute this code? (y/n): ").lower()
                if response != 'y':
                    logger.info("Execution cancelled by user")
                    return result
            
            # Execute all blocks
            logger.info("Executing code blocks...")
            execution_results = self.executor.execute_code_blocks(response)
            result['executions'] = execution_results
            
            # Collect all generated files
            all_files = set()
            for exec_result in execution_results:
                all_files.update(exec_result.get('files', []))
            result['files'] = list(all_files)
            
            # Log summary
            successful = sum(1 for r in execution_results if r['success'])
            logger.info(f"Execution complete: {successful}/{len(execution_results)} successful")
        
        return result
    
    def execute_workflow(
        self,
        tasks: List[Dict[str, Any]],
        share_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a multi-step workflow with Claude.
        
        Args:
            tasks: List of task dicts with keys: prompt, skill_instructions, file_content
            share_context: Whether to share outputs between tasks
            
        Returns:
            List of results from each task
        """
        results = []
        previous_output = None
        
        for i, task in enumerate(tasks, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {i}/{len(tasks)}: {task['prompt'][:50]}...")
            logger.info(f"{'='*60}")
            
            # Build prompt with context from previous task
            prompt = task['prompt']
            if share_context and previous_output:
                prompt = f"""
Previous task output:
{previous_output}

Current task:
{prompt}
"""
            
            # Execute task
            result = self.ask_and_execute(
                prompt=prompt,
                skill_instructions=task.get('skill_instructions'),
                file_content=task.get('file_content'),
                max_tokens=task.get('max_tokens', 4096)
            )
            
            results.append(result)
            
            # Prepare context for next task
            if share_context:
                previous_output = result['response']
                if result['files']:
                    previous_output += f"\n\nGenerated files: {', '.join(result['files'])}"
        
        return results
    
    def _build_prompt(
        self,
        user_prompt: str,
        skill_instructions: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> str:
        """Build comprehensive prompt with all components."""
        parts = []
        
        if skill_instructions:
            parts.append("<skill_instructions>")
            parts.append(skill_instructions)
            parts.append("</skill_instructions>\n")
        
        if file_content:
            parts.append("<input_data>")
            parts.append(file_content)
            parts.append("</input_data>\n")
        
        parts.append("<task>")
        parts.append(user_prompt)
        parts.append("</task>")
        
        return "\n".join(parts)
    
    def get_file(self, filename: str) -> Optional[bytes]:
        """Get content of a generated file."""
        return self.executor.get_file_content(filename)
    
    def save_file(self, filename: str, output_path: str):
        """Save a generated file to a specific location."""
        content = self.get_file(filename)
        if content:
            Path(output_path).write_bytes(content)
            logger.info(f"Saved {filename} to {output_path}")
        else:
            logger.error(f"File {filename} not found")
    
    def list_files(self) -> List[str]:
        """List all generated files."""
        return self.executor.list_generated_files()
    
    def cleanup(self):
        """Clean up executor workspace."""
        self.executor.cleanup()


class SkillBasedAgent(ClaudeCodeAgent):
    """
    Agent specialized for skill-based workflows.
    Automatically loads and applies skill instructions with all assets.

    Production-grade agent for AWS Bedrock Claude skill workflows.
    """

    def __init__(
        self,
        claude_client,
        skills_directory: str = "./custom_skills",
        include_code_guidelines: bool = True,
        **kwargs
    ):
        """
        Initialize skill-based agent.

        Args:
            claude_client: Instance of ClaudeAPI or ClaudeAPIS35
            skills_directory: Directory containing skill folders
            include_code_guidelines: Include code generation guidelines in prompts
            **kwargs: Additional arguments for ClaudeCodeAgent
        """
        super().__init__(claude_client, **kwargs)
        self.skill_loader = SkillLoader(skills_directory)
        self.include_code_guidelines = include_code_guidelines
        self.skills_directory = skills_directory
    
    def load_skill(self, skill_name: str, include_scripts: bool = False) -> Dict[str, any]:
        """
        Load complete skill with all assets.
        
        Args:
            skill_name: Name of skill directory
            include_scripts: Whether to include script contents
            
        Returns:
            Complete skill data dict
        """
        return self.skill_loader.load_skill(skill_name, include_scripts=include_scripts)
    
    def execute_with_skill(
        self,
        skill_name: str,
        prompt: str,
        file_path: Optional[str] = None,
        file_content: Optional[str] = None,
        include_references: bool = True,
        include_scripts: bool = False,
        include_assets: bool = True,
        check_safety: bool = True,
        stop_on_error: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a task using a specific skill with all its assets.

        Args:
            skill_name: Name of skill to use
            prompt: User prompt
            file_path: Optional path to input file
            file_content: Optional file content (alternative to file_path)
            include_references: Include skill reference files
            include_scripts: Include skill scripts
            include_assets: Include skill asset contents
            check_safety: Perform safety checks on code
            stop_on_error: Stop execution if a code block fails
            **kwargs: Additional arguments for ask_and_execute

        Returns:
            Execution result with keys: success, response, code_blocks, executions, files
        """
        start_time = time.time()

        # Load file content if file_path provided
        if file_path and not file_content:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                # Try with latin-1 as fallback
                with open(file_path, 'r', encoding='latin-1') as f:
                    file_content = f.read()

        # Build comprehensive prompt with skill loader
        full_prompt = self.skill_loader.build_comprehensive_prompt(
            skill_name=skill_name,
            user_query=prompt,
            file_content=file_content,
            include_references=include_references,
            include_scripts=include_scripts,
            include_assets=include_assets,
            include_code_guidelines=self.include_code_guidelines,
            workspace_dir=self.executor.workspace_dir
        )

        # Get Claude's response
        logger.info("Sending request to Claude with skill instructions...")
        try:
            response = self.claude_client.invoke_llm_model(
                prompt=full_prompt,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.0)
            )
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'code_blocks': [],
                'executions': [],
                'files': [],
                'elapsed_time': time.time() - start_time
            }

        logger.info(f"Received response ({len(response)} chars)")

        # Store in history
        self.conversation_history.append({
            'skill': skill_name,
            'prompt': prompt,
            'response': response,
            'timestamp': time.time()
        })

        # Extract code blocks
        code_blocks = self.executor.extract_python_blocks(response)

        result = {
            'success': True,
            'response': response,
            'code_blocks': code_blocks,
            'executions': [],
            'files': [],
            'skill': skill_name,
            'elapsed_time': 0
        }

        # Execute code if requested
        should_execute = kwargs.get('execute', self.auto_execute)

        if should_execute and code_blocks:
            logger.info(f"Found {len(code_blocks)} code blocks")

            if self.require_confirmation:
                print("\n" + "=" * 60)
                print("CODE FOUND IN RESPONSE")
                print("=" * 60)
                for i, code in enumerate(code_blocks, 1):
                    print(f"\nBlock {i}:")
                    print("-" * 40)
                    print(code[:300] + ("..." if len(code) > 300 else ""))
                print("\n" + "=" * 60)

                user_response = input("Execute this code? (y/n): ").lower()
                if user_response != 'y':
                    logger.info("Execution cancelled by user")
                    result['elapsed_time'] = time.time() - start_time
                    return result

            # Execute all blocks with the configured safety level
            logger.info("Executing code blocks...")
            execution_results = self.executor.execute_code_blocks(
                response,
                check_safety=check_safety,
                stop_on_error=stop_on_error
            )
            result['executions'] = execution_results

            # Collect all generated files
            all_files = set()
            for exec_result in execution_results:
                all_files.update(exec_result.get('files', []))
            result['files'] = list(all_files)

            # Log summary
            successful = sum(1 for r in execution_results if r['success'])
            logger.info(f"Execution complete: {successful}/{len(execution_results)} successful")

            # Update overall success status
            if execution_results and not any(r['success'] for r in execution_results):
                result['success'] = False

        result['elapsed_time'] = time.time() - start_time
        return result
    
    def list_skills(self) -> List[str]:
        """List available skills."""
        return self.skill_loader.list_skills()

    def get_skill_info(self, skill_name: str) -> Dict[str, Any]:
        """Get detailed information about a skill."""
        return self.skill_loader.get_skill_info(skill_name)

    def validate_skill(self, skill_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a skill's structure and configuration.

        Args:
            skill_name: Name of skill to validate

        Returns:
            (is_valid, list_of_issues)
        """
        return self.skill_loader.validate_skill(skill_name)

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all executions in the current session.

        Returns:
            Dict with execution statistics
        """
        total_executions = 0
        successful_executions = 0
        failed_executions = 0
        total_files = set()

        for entry in self.conversation_history:
            if 'executions' in entry:
                for exec_result in entry.get('executions', []):
                    total_executions += 1
                    if exec_result.get('success'):
                        successful_executions += 1
                    else:
                        failed_executions += 1
                    total_files.update(exec_result.get('files', []))

        return {
            'total_conversations': len(self.conversation_history),
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'total_files_generated': len(total_files),
            'files': list(total_files)
        }

    def reset_session(self):
        """Reset the agent session, clearing history and accumulated code."""
        self.conversation_history = []
        self.executor.reset_accumulated()
        logger.info("Session reset complete")


# Example usage
if __name__ == "__main__":
    from your_module import ClaudeAPI  # Update with your actual module
    import os
    
    # Initialize Claude client
    client = ClaudeAPI(
        region_name='eu-west-1',
        profile_arn=os.getenv("CLAUDE_INFERENCE_PROFILE_ARN")
    )
    
    # Example 1: Basic agent usage
    print("=" * 60)
    print("Example 1: Basic Code Agent")
    print("=" * 60)
    
    agent = ClaudeCodeAgent(
        claude_client=client,
        auto_execute=True,
        require_confirmation=True
    )
    
    result = agent.ask_and_execute(
        prompt="Create a Python script that generates a bar chart showing monthly sales data. Use random data for demonstration."
    )
    
    print(f"\nSuccess: {result['success']}")
    print(f"Generated files: {result['files']}")
    
    # Example 2: Skill-based agent
    print("\n" + "=" * 60)
    print("Example 2: Skill-Based Agent")
    print("=" * 60)
    
    skill_agent = SkillBasedAgent(
        claude_client=client,
        skills_directory="./custom_skills",
        auto_execute=True,
        require_confirmation=False
    )
    
    # List available skills
    skills = skill_agent.list_skills()
    print(f"Available skills: {skills}")
    
    # Use a skill
    if skills:
        result = skill_agent.execute_with_skill(
            skill_name=skills[0],
            prompt="Process this data according to the skill instructions",
            file_path="./data/sample.csv"
        )
        
        print(f"Files generated: {result['files']}")
    
    # Example 3: Multi-step workflow
    print("\n" + "=" * 60)
    print("Example 3: Multi-Step Workflow")
    print("=" * 60)
    
    workflow_agent = ClaudeCodeAgent(
        claude_client=client,
        persistent=True,
        workspace_dir="./workflow_workspace"
    )
    
    tasks = [
        {
            'prompt': 'Generate synthetic time series data with trend and seasonality. Save to CSV.',
            'max_tokens': 2000
        },
        {
            'prompt': 'Load the generated CSV and create visualizations showing trend, seasonality, and residuals.',
            'max_tokens': 3000
        },
        {
            'prompt': 'Analyze the time series and provide statistical summary.',
            'max_tokens': 2000
        }
    ]
    
    results = workflow_agent.execute_workflow(tasks, share_context=True)
    
    print(f"\nWorkflow completed: {len(results)} tasks")
    for i, result in enumerate(results, 1):
        print(f"Task {i}: {len(result.get('files', []))} files generated")
