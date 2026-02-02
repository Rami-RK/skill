# Automated Code Execution Framework for Claude + AWS Bedrock

A comprehensive framework for automating code execution with AWS Bedrock Claude API, providing secure sandboxing, dependency management, and workflow orchestration.

## Features

✅ **Secure Code Execution**
- Sandboxed execution environment
- Configurable timeout and memory limits
- Import whitelisting/blacklisting
- Automatic cleanup

✅ **Automatic Dependency Management**
- Auto-detection of required packages
- Automatic installation with pip
- Package caching to avoid reinstalls

✅ **Smart Code Extraction**
- Extracts Python code from Claude responses
- Supports markdown code blocks
- Handles multiple code blocks

✅ **Workflow Orchestration**
- Multi-step task execution
- Context sharing between steps
- Persistent workspace support

✅ **Skill-Based Execution**
- Load and apply skill instructions
- Skill library management
- Automatic skill-prompt integration

## Installation

### Prerequisites

```bash
# Python 3.8+
pip install boto3 python-dotenv --break-system-packages

# Optional but recommended for data analysis
pip install pandas numpy matplotlib seaborn scikit-learn --break-system-packages
```

### Setup

1. **Clone or download the files:**
   - `code_executor.py` - Core execution framework
   - `claude_code_integration.py` - Claude API integration
   - `core_bedrock.ipynb` - Basic usage examples
   - `automated_execution_demo.ipynb` - Advanced examples

2. **Configure environment:**

```bash
# Create .env file
echo "CLAUDE_INFERENCE_PROFILE_ARN=your_arn_here" > .env
```

3. **Update imports:**

In both notebooks and `claude_code_integration.py`, update:
```python
from your_module import ClaudeAPI
```
to match your actual module name.

## Quick Start

### Basic Execution

```python
from code_executor import CodeExecutor

# Create executor
executor = CodeExecutor(timeout=60, auto_install_packages=True)

# Execute code
code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
df.to_csv('output.csv')
"""

result = executor.execute_python_code(code)
print(f"Success: {result['success']}")
print(f"Output: {result['output']}")
print(f"Files: {result['files']}")

executor.cleanup()
```

### With Claude Integration

```python
from claude_code_integration import ClaudeCodeAgent
from your_module import ClaudeAPI

# Initialize
client = ClaudeAPI(region_name='eu-west-1')
agent = ClaudeCodeAgent(
    claude_client=client,
    auto_execute=True,
    require_confirmation=True
)

# Ask Claude and automatically execute generated code
result = agent.ask_and_execute(
    prompt="Create a bar chart showing sales by quarter"
)

print(f"Generated files: {result['files']}")
```

### Skill-Based Workflow

```python
from claude_code_integration import SkillBasedAgent

# Initialize skill agent
skill_agent = SkillBasedAgent(
    claude_client=client,
    skills_directory="./custom_skills"
)

# List available skills
print(skill_agent.list_skills())

# Execute with skill
result = skill_agent.execute_with_skill(
    skill_name="analyzing-time-series",
    prompt="Analyze this data",
    file_path="./data/sales.csv"
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              ClaudeCodeAgent / SkillBasedAgent               │
│  - Prompt building                                           │
│  - Response parsing                                          │
│  - Workflow orchestration                                    │
└─────────────────────────────────────────────────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────┐          ┌──────────────────────────┐
│   Claude API         │          │    CodeExecutor          │
│   (AWS Bedrock)      │          │  - Code extraction       │
│                      │          │  - Safety checks         │
│  - Generate code     │          │  - Package management    │
│  - Provide analysis  │          │  - Sandboxed execution   │
└──────────────────────┘          └──────────────────────────┘
                                              │
                                              ▼
                                  ┌──────────────────────────┐
                                  │  Isolated Workspace      │
                                  │  - File management       │
                                  │  - Output capture        │
                                  └──────────────────────────┘
```

## API Reference

### CodeExecutor

```python
CodeExecutor(
    workspace_dir: Optional[str] = None,
    timeout: int = 300,
    max_memory_mb: int = 1024,
    allowed_imports: Optional[List[str]] = None,
    blocked_imports: Optional[List[str]] = None,
    auto_install_packages: bool = True
)
```

**Methods:**
- `execute_python_code(code: str) -> Dict` - Execute Python code
- `execute_code_blocks(text: str) -> List[Dict]` - Extract and execute all code blocks
- `list_generated_files() -> List[str]` - List generated files
- `get_file_content(filename: str) -> bytes` - Get file content
- `cleanup()` - Clean up workspace

### ClaudeCodeAgent

```python
ClaudeCodeAgent(
    claude_client,
    workspace_dir: Optional[str] = None,
    auto_execute: bool = True,
    require_confirmation: bool = True,
    persistent: bool = False,
    **executor_kwargs
)
```

**Methods:**
- `ask_and_execute(prompt: str, ...) -> Dict` - Send prompt and execute code
- `execute_workflow(tasks: List[Dict], ...) -> List[Dict]` - Multi-step workflow
- `list_files() -> List[str]` - List generated files
- `save_file(filename: str, output_path: str)` - Save file to location
- `cleanup()` - Clean up workspace

### SkillBasedAgent

```python
SkillBasedAgent(
    claude_client,
    skills_directory: str = "./custom_skills",
    **kwargs
)
```

**Methods:**
- `load_skill(skill_name: str) -> str` - Load skill instructions
- `execute_with_skill(skill_name: str, prompt: str, ...) -> Dict` - Execute with skill
- `list_skills() -> List[str]` - List available skills

## Security Features

### Import Control

```python
executor = CodeExecutor(
    # Whitelist only specific packages
    allowed_imports=['pandas', 'numpy', 'matplotlib'],
    
    # Blacklist dangerous operations
    blocked_imports=['os.system', 'subprocess', 'eval', 'exec']
)
```

### Execution Limits

```python
executor = CodeExecutor(
    timeout=300,  # Maximum 5 minutes
    max_memory_mb=1024  # Maximum 1GB RAM
)
```

### Safe Execution

- Code runs in subprocess for isolation
- Workspace directory isolated from system
- Automatic cleanup prevents file accumulation
- Signal-based timeout handling

## Use Cases

### 1. Data Analysis Pipeline

```python
workflow = [
    {
        'prompt': 'Load CSV and perform EDA',
        'file_path': 'data.csv'
    },
    {
        'prompt': 'Create visualizations of key metrics',
    },
    {
        'prompt': 'Generate statistical summary report',
    }
]

results = agent.execute_workflow(workflow, share_context=True)
```

### 2. Document Generation

```python
result = agent.ask_and_execute(
    prompt="""
    Create a Word document report with:
    1. Executive summary
    2. Data analysis from previous step
    3. Visualizations
    4. Recommendations
    """,
    skill_instructions=docx_skill
)
```

### 3. Automated Testing

```python
result = agent.ask_and_execute(
    prompt="Generate unit tests for the function in this file",
    file_content=function_code
)
```

### 4. Code Generation & Execution

```python
agent = ClaudeCodeAgent(client, require_confirmation=False)

result = agent.ask_and_execute(
    prompt="Create a web scraper for product prices and save to database"
)
```

## Workflow Patterns

### Pattern 1: Sequential Processing

```python
tasks = [
    {'prompt': 'Load and clean data'},
    {'prompt': 'Perform analysis'},
    {'prompt': 'Create visualizations'},
    {'prompt': 'Generate report'}
]

results = agent.execute_workflow(tasks, share_context=True)
```

### Pattern 2: Iterative Refinement

```python
# Step 1: Generate initial code
result1 = agent.ask_and_execute("Create data visualization")

# Step 2: Refine based on output
result2 = agent.ask_and_execute(
    f"Improve this visualization: {result1['response']}"
)
```

### Pattern 3: Skill Chaining

```python
# Analyze with one skill
result1 = skill_agent.execute_with_skill(
    skill_name="analyzing-time-series",
    prompt="Analyze data",
    file_path="data.csv"
)

# Generate questions with another skill
result2 = skill_agent.execute_with_skill(
    skill_name="generating-practice-questions",
    prompt=f"Create quiz from analysis: {result1['response']}"
)
```

## Configuration Examples

### Development Mode
```python
dev_agent = ClaudeCodeAgent(
    client,
    auto_execute=False,  # Review before execution
    require_confirmation=True,
    timeout=60
)
```

### Production Mode
```python
prod_agent = ClaudeCodeAgent(
    client,
    auto_execute=True,
    require_confirmation=False,
    persistent=True,
    workspace_dir="/var/app/workspace",
    timeout=600,
    allowed_imports=['pandas', 'numpy', 'matplotlib', 'seaborn']
)
```

### Restricted Mode
```python
restricted_agent = ClaudeCodeAgent(
    client,
    auto_execute=True,
    allowed_imports=['pandas', 'numpy'],  # Very limited
    blocked_imports=['os', 'sys', 'subprocess', 'socket'],
    timeout=120,
    max_memory_mb=512
)
```

## Troubleshooting

### Package Installation Fails

```python
# Manual installation
import subprocess
subprocess.run([
    "pip", "install", "package_name", "--break-system-packages"
])

# Or disable auto-install
executor = CodeExecutor(auto_install_packages=False)
```

### Timeout Issues

```python
# Increase timeout
executor = CodeExecutor(timeout=600)  # 10 minutes

# Or split into smaller tasks
workflow = [
    {'prompt': 'Step 1 (fast)', 'max_tokens': 1000},
    {'prompt': 'Step 2 (fast)', 'max_tokens': 1000},
]
```

### Import Errors

```python
# Check what imports are needed
packages = executor.extract_required_packages(code)
print(f"Required: {packages}")

# Add to whitelist
executor = CodeExecutor(
    allowed_imports=packages + ['your', 'extra', 'packages']
)
```

### Memory Issues

```python
# Increase memory limit
executor = CodeExecutor(max_memory_mb=2048)

# Or process data in chunks
prompt = "Process data in 1000-row chunks to save memory"
```

## Best Practices

1. **Always Review Critical Code**
   ```python
   agent = ClaudeCodeAgent(client, require_confirmation=True)
   ```

2. **Use Persistent Workspaces for Related Tasks**
   ```python
   agent = ClaudeCodeAgent(client, persistent=True)
   ```

3. **Set Appropriate Timeouts**
   ```python
   # Short for simple tasks
   executor = CodeExecutor(timeout=30)
   
   # Long for complex analysis
   executor = CodeExecutor(timeout=600)
   ```

4. **Leverage Skills for Consistency**
   ```python
   skill_agent = SkillBasedAgent(client, skills_directory="./skills")
   ```

5. **Handle Errors Gracefully**
   ```python
   result = agent.ask_and_execute(prompt)
   if not result['success']:
       print(f"Error: {result['error']}")
       # Retry or handle error
   ```

6. **Clean Up Resources**
   ```python
   try:
       result = agent.ask_and_execute(prompt)
   finally:
       agent.cleanup()
   ```

## Examples Directory Structure

```
project/
├── code_executor.py
├── claude_code_integration.py
├── core_bedrock.ipynb
├── automated_execution_demo.ipynb
├── custom_skills/
│   ├── analyzing-time-series/
│   │   └── SKILL.md
│   ├── generating-practice-questions/
│   │   └── SKILL.md
│   └── ...
├── data/
│   ├── retail_sales.csv
│   └── ...
└── outputs/
    └── (generated files)
```

## Contributing

Feel free to extend the framework:

1. Add new safety checks in `CodeExecutor.check_imports()`
2. Create custom agents by subclassing `ClaudeCodeAgent`
3. Build domain-specific executors
4. Add support for other languages (bash, R, etc.)

## License

MIT License - use freely in your projects!

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example notebooks
3. Examine execution logs for details

---

**Note**: This framework provides security features but is not a substitute for careful code review, especially in production environments. Always validate generated code before execution in critical systems.
