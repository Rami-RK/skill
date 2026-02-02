# bedrock_agent_skills - Production-Grade Skill Execution for AWS Bedrock Claude

A comprehensive, production-ready framework for executing Claude-generated code with AWS Bedrock, providing secure sandboxing, skill management, and workflow orchestration.

## Features

### Core Capabilities
- **Secure Code Execution** with configurable security levels
- **UTF-8 Encoding Support** for cross-platform compatibility
- **Accumulated Execution Mode** for dependent code blocks
- **Flexible Skill Loading** supporting multiple directory structures
- **Automatic Dependency Management** with package detection

### Security Levels
- `strict`: Blocks subprocess, eval, exec (maximum safety)
- `moderate`: Allows subprocess for skill scripts, blocks eval/exec (recommended)
- `permissive`: Minimal restrictions (use with caution)

### What's New in v1.0.0
- Fixed Unicode encoding errors on Windows
- Added configurable security levels
- Accumulated code block execution for context preservation
- Support for both flat and nested skill directory structures
- Enhanced prompts with code generation guidelines
- Improved error handling and logging

## Installation

### Prerequisites

```bash
# Python 3.8+
pip install boto3 python-dotenv pandas numpy matplotlib

# Optional for document generation
pip install python-docx openpyxl
```

### Setup

```python
from bedrock_agent_skills import SkillBasedAgent, create_agent
from claude_client import ClaudeAPI  # Your Bedrock client

# Method 1: Using factory function (recommended)
agent = create_agent(
    claude_client=ClaudeAPI(profile_arn="your-arn"),
    skills_directory="./custom_skills",
    security_level='moderate'
)

# Method 2: Direct instantiation
agent = SkillBasedAgent(
    claude_client=ClaudeAPI(profile_arn="your-arn"),
    skills_directory="./custom_skills",
    security_level='moderate',
    accumulate_blocks=True,
    auto_execute=True,
    require_confirmation=False,
    persistent=True,
    workspace_dir="./workspace"
)
```

## Quick Start

### Basic Skill Execution

```python
from bedrock_agent_skills import SkillBasedAgent
from claude_client import ClaudeAPI

# Initialize
client = ClaudeAPI(profile_arn="your-arn")
agent = SkillBasedAgent(
    claude_client=client,
    skills_directory="./custom_skills",
    security_level='moderate',
    auto_execute=True,
    require_confirmation=False
)

# List available skills
skills = agent.list_skills()
print(f"Available skills: {skills}")

# Execute with a skill
result = agent.execute_with_skill(
    skill_name="generating-practice-questions",
    prompt="Generate practice questions from these lecture notes",
    file_path="./lecture_notes/notes04.tex",
    include_references=True,
    include_scripts=False,
    max_tokens=4096
)

# Check results
print(f"Success: {result['success']}")
print(f"Code blocks found: {len(result['code_blocks'])}")
print(f"Files generated: {result['files']}")
```

### Time Series Analysis Example

```python
result = agent.execute_with_skill(
    skill_name="analyzing-time-series",
    prompt="""
    Analyze this time series data and create:
    1. Diagnostic plots (trend, seasonality, residuals)
    2. Statistical analysis
    3. A summary report
    """,
    file_path="./data/retail_sales.csv",
    include_references=True,
    include_scripts=True,  # Include helper scripts for complex tasks
    max_tokens=16384
)
```

### Multi-Step Workflow

```python
from bedrock_agent_skills import ClaudeCodeAgent

workflow_agent = ClaudeCodeAgent(
    claude_client=client,
    persistent=True,
    workspace_dir="./workflow_workspace",
    auto_execute=True,
    require_confirmation=False
)

tasks = [
    {
        'prompt': 'Load ./data/retail_sales.csv and perform initial analysis. Save findings to findings.txt',
        'max_tokens': 6000
    },
    {
        'prompt': 'Create visualizations based on the analysis. Save as PNG files.',
        'max_tokens': 4000
    },
    {
        'prompt': 'Generate a comprehensive report summarizing findings and visualizations.',
        'max_tokens': 5000
    }
]

results = workflow_agent.execute_workflow(tasks, share_context=True)
```

## Skill Directory Structure

The framework supports two directory structures:

### Flat Structure (Simple)
```
custom_skills/
└── your-skill/
    ├── SKILL.md           # Required: Main instructions
    ├── reference.md       # Optional: Additional documentation
    └── assets/
        └── template.md    # Optional: Templates
```

### Nested Structure (L5 Style)
```
custom_skills/
└── your-skill/
    ├── SKILL.md           # Required: Main instructions
    ├── references/
    │   └── examples.md    # Reference documentation
    ├── scripts/
    │   ├── __init__.py
    │   └── helper.py      # Helper scripts
    └── assets/
        └── template.md
```

### SKILL.md Format

```markdown
---
name: your-skill-name
description: Brief description of what this skill does
license: MIT
---

# Skill Name

## Overview
What this skill does...

## Instructions
Step-by-step guidance for the AI...

## Examples
Code and usage examples...

## Best Practices
Tips and recommendations...
```

## API Reference

### SkillBasedAgent

```python
SkillBasedAgent(
    claude_client,                          # Your Claude API client
    skills_directory: str = "./custom_skills",
    security_level: str = 'moderate',       # 'strict', 'moderate', 'permissive'
    accumulate_blocks: bool = True,         # Preserve context between blocks
    auto_execute: bool = True,              # Auto-execute generated code
    require_confirmation: bool = True,      # Ask before execution
    persistent: bool = False,               # Persistent workspace
    workspace_dir: str = None,              # Custom workspace path
    timeout: int = 300,                     # Execution timeout (seconds)
    encoding: str = 'utf-8'                 # File encoding
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `execute_with_skill(...)` | Execute task with skill instructions |
| `list_skills()` | List available skills |
| `get_skill_info(name)` | Get skill metadata |
| `validate_skill(name)` | Validate skill structure |
| `list_files()` | List generated files |
| `get_file(name)` | Get file content |
| `reset_session()` | Clear history and accumulated code |
| `get_execution_summary()` | Get execution statistics |

### CodeExecutor

```python
CodeExecutor(
    workspace_dir: str = None,
    timeout: int = 300,
    security_level: str = 'moderate',
    accumulate_blocks: bool = True,
    encoding: str = 'utf-8',
    auto_install_packages: bool = True
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `execute_python_code(code)` | Execute Python code |
| `execute_code_blocks(text)` | Extract and execute all code blocks |
| `sanitize_code(code)` | Remove problematic Unicode characters |
| `check_imports(code)` | Check code safety |
| `reset_accumulated()` | Clear accumulated context |
| `list_generated_files()` | List workspace files |
| `get_file_content(name)` | Get file content |
| `cleanup()` | Clean up workspace |

### SkillLoader

```python
SkillLoader(skills_directory: str = "./custom_skills")
```

**Methods:**

| Method | Description |
|--------|-------------|
| `load_skill(name, include_scripts)` | Load complete skill data |
| `build_comprehensive_prompt(...)` | Build prompt with skill context |
| `list_skills()` | List available skills |
| `get_skill_info(name)` | Get skill metadata |
| `validate_skill(name)` | Validate skill structure |
| `get_skill_asset(name, path)` | Get asset file content |

## Security Configuration

### Security Levels Explained

```python
# Strict: Maximum safety, blocks many operations
agent = SkillBasedAgent(client, security_level='strict')
# Blocked: subprocess, os.system, eval, exec, compile, __import__, ctypes, socket

# Moderate: Balanced (recommended for skills)
agent = SkillBasedAgent(client, security_level='moderate')
# Blocked: ctypes, socket, eval, exec

# Permissive: Minimal restrictions
agent = SkillBasedAgent(client, security_level='permissive')
# Blocked: None (use with caution)
```

### Custom Security Configuration

```python
from bedrock_agent_skills import CodeExecutor

executor = CodeExecutor(
    security_level='moderate',
    blocked_imports=['socket', 'ctypes', 'pickle'],  # Override defaults
)
```

## Troubleshooting

### Unicode Encoding Errors

**Problem:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution:** The framework now automatically sanitizes Unicode characters. If issues persist:

```python
# The sanitize_code method handles this automatically
agent = SkillBasedAgent(client, encoding='utf-8')
```

### Code Blocks Not Self-Contained

**Problem:** `NameError: name 'np' is not defined`

**Solution:** Enable accumulated execution mode (default):

```python
agent = SkillBasedAgent(
    client,
    accumulate_blocks=True  # Preserves context between blocks
)
```

Or add explicit instructions to your prompt:

```python
result = agent.execute_with_skill(
    skill_name="your-skill",
    prompt="Each code block must be fully self-contained with all imports. ...",
    ...
)
```

### Safety Check Blocking subprocess

**Problem:** `Safety check failed: Blocked import detected: 'subprocess'`

**Solution:** Use 'moderate' or 'permissive' security level:

```python
agent = SkillBasedAgent(
    client,
    security_level='moderate'  # Allows subprocess
)
```

### Package Installation Fails

**Problem:** Package installation timeout or failure

**Solution:**

```python
# Pre-install packages manually
import subprocess
subprocess.run(["pip", "install", "package_name", "--break-system-packages"])

# Or disable auto-install
from bedrock_agent_skills import CodeExecutor
executor = CodeExecutor(auto_install_packages=False)
```

## Best Practices

### 1. Use Appropriate Security Level

```python
# For trusted skill execution
agent = SkillBasedAgent(client, security_level='moderate')

# For untrusted or external code
agent = SkillBasedAgent(client, security_level='strict')
```

### 2. Enable Accumulated Execution for Complex Tasks

```python
agent = SkillBasedAgent(
    client,
    accumulate_blocks=True  # Helps with dependent code blocks
)
```

### 3. Include References for Advanced Features

```python
result = agent.execute_with_skill(
    skill_name="analyzing-time-series",
    include_references=True,   # Include skill documentation
    include_scripts=True,      # Include helper scripts
    ...
)
```

### 4. Use Persistent Workspaces for Multi-Step Tasks

```python
agent = SkillBasedAgent(
    client,
    persistent=True,
    workspace_dir="./my_workspace"
)
```

### 5. Handle Errors Gracefully

```python
result = agent.execute_with_skill(...)

if not result['success']:
    print(f"Error: {result.get('error')}")
    for exec_result in result['executions']:
        if not exec_result['success']:
            print(f"Block {exec_result['block_number']} failed: {exec_result['error']}")
```

## Comparison: Anthropic API vs This Implementation

| Feature | Anthropic Skills API | bedrock_agent_skills |
|---------|---------------------|---------------------|
| Skill Upload | API-based | Local directory |
| File Upload | API-based | Local file loading |
| Code Execution | Managed sandbox | Local subprocess |
| File Download | API-based | Direct file access |
| Asset Loading | Automatic | Controlled via flags |
| Security | Platform-managed | Configurable levels |
| Cost | Per-request | AWS Bedrock pricing |

## Examples

See the following notebooks for complete examples:

- `core_bedrock_updated.ipynb` - Main usage examples
- `automated_execution_demo.ipynb` - Advanced automation patterns

## Contributing

Contributions welcome! Areas for improvement:

1. Add support for additional languages (R, bash, etc.)
2. Implement more security checks
3. Add caching for skill prompts
4. Create additional skill templates

## License

MIT License - use freely in your projects.

---

**Note:** This framework provides security features but is not a substitute for careful code review. Always validate generated code before execution in production systems.
