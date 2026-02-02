# Skills with AWS Bedrock Claude

A production-grade implementation of Anthropic's Skills API pattern for AWS Bedrock Claude. This project enables you to use custom skills with Claude on AWS Bedrock, providing secure code execution, skill management, and multi-step workflow orchestration.

> **Note**: This is an AWS Bedrock adaptation of the [Anthropic Skills API](https://platform.claude.com/docs/en/build-with-claude/skills-guide). While Anthropic's native Skills API runs code in a managed sandbox, this implementation executes code locally via subprocess with configurable security controls.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Creating Custom Skills](#creating-custom-skills)
- [API Reference](#api-reference)
- [Security Levels](#security-levels)
- [Troubleshooting](#troubleshooting)
- [Key Differences from Anthropic Skills API](#key-differences-from-anthropic-skills-api)
- [Estimated Costs](#estimated-costs)
- [Additional Resources](#additional-resources)

---

## Features

| Feature | Description |
|---------|-------------|
| **Skill-Based Execution** | Load and apply custom skill instructions to Claude prompts |
| **Secure Code Execution** | Configurable security levels (strict, moderate, permissive) |
| **Multi-Step Workflows** | Chain multiple tasks with context sharing between steps |
| **Auto Dependencies** | Automatically detects and installs required Python packages |
| **File Generation** | Generate plots (PNG), documents (DOCX), JSON, and more |
| **UTF-8 Support** | Cross-platform compatibility (Windows/Linux/macOS) |
| **Accumulated Execution** | Maintain context between code blocks |

---

## Project Structure

```
skill/
|
|-- bedrock_agent_skills/
|   |-- __init__.py                  # Package exports and version info
|   |-- claude_code_integration.py   # ClaudeCodeAgent and SkillBasedAgent
|   |-- code_executor.py             # Secure code execution engine
|   |-- skill_loader.py              # Skill loading and prompt building
|   |-- exceptions.py                # Custom exception classes
|   +-- README.md                    # Module documentation
|
|-- custom_skills/
|   |
|   |-- analyzing-time-series/
|   |   |-- SKILL.md                 # Main skill instructions
|   |   |-- interpretation.md        # Reference: result interpretation
|   |   +-- scripts/
|   |       |-- __init__.py
|   |       |-- diagnose.py          # Diagnostic analysis
|   |       |-- visualize.py         # Visualization utilities
|   |       +-- ts_utils.py          # Time series utilities
|   |
|   +-- generating-practice-questions/
|       |-- SKILL.md                 # Main skill instructions
|       |-- examples_by_topic.md     # Reference: topic examples
|       +-- assets/
|           |-- markdown_template.md
|           +-- questions_template.tex
|
|-- data/
|   |-- retail_sales.csv             # Time series dataset
|   +-- city_data.csv                # Additional sample data
|
|-- lecture_notes/
|   +-- notes04.tex                  # LaTeX lecture notes (Part 1 input)
|
|-- workspace/                       # Output directory (Parts 1-2)
|-- workflow_workspace/              # Output directory (Part 4)
|-- sample_outputs/                  # Pre-generated example outputs
|
|-- claude_client.py                 # AWS Bedrock Claude API client
|-- core_bedrock_updated.ipynb       # Main demonstration notebook
|-- requirements.txt                 # Python dependencies
|-- .env                             # Environment variables (create this)
|-- .gitignore                       # Git ignore rules
+-- README.md                        # This file
```

### Key Files

| File | Purpose |
|------|---------|
| claude_client.py | AWS Bedrock client with retry logic and error handling |
| core_bedrock_updated.ipynb | Main notebook with 4 demonstration parts |
| bedrock_agent_skills/ | Reusable framework module for skill execution |
| custom_skills/ | Your custom skill definitions (add your own here) |

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **AWS Account** with Bedrock access enabled
- **Claude Model Access** on AWS Bedrock (Claude 3.5 or 4.5 Sonnet)
- **AWS Credentials** configured (AWS CLI, env vars, or IAM role)
- **VS Code** or **Jupyter Notebook** for running the notebook

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rami-RK/skill.git
cd skill
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

| Platform | Command |
|----------|---------|
| Windows (PowerShell) | .\venv\Scripts\Activate.ps1 |
| Windows (CMD) | venv\Scripts\activate.bat |
| macOS / Linux | source venv/bin/activate |

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Included packages:**
- python-dotenv - Environment variable management
- pandas - Data manipulation
- boto3 - AWS SDK for Python
- requests - HTTP library
- python-docx - Word document generation
- numpy, matplotlib, statsmodels - Data analysis and visualization

### Step 4: Configure AWS Credentials

**Option A: AWS CLI (Recommended)**

```bash
aws configure
```

**Option B: Environment Variables**

Windows PowerShell:
```powershell
$env:AWS_ACCESS_KEY_ID = "your_access_key"
$env:AWS_SECRET_ACCESS_KEY = "your_secret_key"
$env:AWS_DEFAULT_REGION = "eu-west-1"
```

Linux/macOS:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=eu-west-1
```

---

## Configuration

### Create the .env File

Create a .env file in the project root:

```env
# AWS Bedrock Claude Inference Profile ARN
CLAUDE_45_INFERENCE_PROFILE_ARN=arn:aws:bedrock:eu-west-1:YOUR_ACCOUNT_ID:inference-profile/YOUR_PROFILE_ID

# Optional: For Claude 3.5 Sonnet
CLAUDE_INFERENCE_PROFILE_ARN=arn:aws:bedrock:eu-west-1:YOUR_ACCOUNT_ID:inference-profile/YOUR_PROFILE_ID
```

### Getting Your Inference Profile ARN

1. Sign in to **AWS Console** then **Amazon Bedrock**
2. Navigate to **Cross-region inference** or **Inference profiles**
3. Create or select an inference profile for Claude
4. Copy the **ARN** (format: arn:aws:bedrock:REGION:ACCOUNT_ID:inference-profile/PROFILE_ID)

> **Important**: Ensure your AWS account has access to Claude models. Request model access in the Bedrock console if needed.

---

## Quick Start

### Option 1: Run the Notebook (Recommended)

```bash
# Open in VS Code
code core_bedrock_updated.ipynb

# Or in Jupyter
jupyter notebook core_bedrock_updated.ipynb
```

### Option 2: Python Script

```python
from dotenv import load_dotenv
import os
from claude_client import ClaudeAPI
from bedrock_agent_skills import SkillBasedAgent

# Load environment
load_dotenv()

# Initialize client
client = ClaudeAPI(profile_arn=os.getenv("CLAUDE_45_INFERENCE_PROFILE_ARN"))

# Create agent
agent = SkillBasedAgent(
    claude_client=client,
    skills_directory="./custom_skills",
    security_level='moderate',
    auto_execute=True,
    workspace_dir="./workspace"
)

# Execute with skill
result = agent.execute_with_skill(
    skill_name="analyzing-time-series",
    prompt="Analyze this data and create visualizations",
    file_path="./data/retail_sales.csv",
    include_references=True
)

print(f"Success: {result['success']}")
print(f"Files: {result['files']}")
```

---

## Usage Guide

The notebook demonstrates four key use cases:

### Part 1: Practice Question Generator

| Item | Details |
|------|---------|
| **Skill** | generating-practice-questions |
| **Input** | lecture_notes/notes04.tex |
| **Output** | practice_questions_ml_models.md |
| **Purpose** | Generate practice questions from lecture notes |

### Part 2: Time Series Analysis

| Item | Details |
|------|---------|
| **Skill** | analyzing-time-series |
| **Input** | data/retail_sales.csv |
| **Outputs** | 5 PNG plots + Word document report |
| **Purpose** | Comprehensive time series analysis |

**Generated Files:**
- workspace/plots/timeseries.png
- workspace/plots/histogram.png
- workspace/plots/rolling_stats.png
- workspace/plots/acf_pacf.png
- workspace/plots/decomposition.png
- workspace/time_series_report.docx

### Part 3: Skill Asset Loading

Understanding how skill assets affect prompts:

| Strategy | Includes | Token Usage | Use Case |
|----------|----------|-------------|----------|
| Minimal | SKILL.md only | ~3,000 | Simple tasks |
| With refs | + reference .md files | ~8,000 | Standard tasks |
| Full | + Python scripts | ~15,000 | Complex tasks |

### Part 4: Multi-Step Workflows

| Step | Task | Output |
|------|------|--------|
| 1 | Analyze data | findings.txt, diagnostics.json |
| 2 | Create plots | 4 PNG visualizations |
| 3 | Generate report | workflow_report.docx |

---

## Creating Custom Skills

### Directory Structure

```
custom_skills/
+-- your-skill-name/
    |-- SKILL.md              # Required: Main instructions
    |-- reference1.md         # Optional: Additional docs
    |-- scripts/              # Optional: Helper scripts
    |   +-- helper.py
    +-- assets/               # Optional: Templates
        +-- template.docx
```

### SKILL.md Format

```markdown
---
name: your-skill-name
description: Brief description
license: MIT
---

# Skill Name

## Overview
What this skill does.

## Input Format
Expected input format.

## Workflow
Step-by-step instructions.

## Examples
Code examples.
```

---

## API Reference

### SkillBasedAgent

```python
SkillBasedAgent(
    claude_client,                      # ClaudeAPI instance
    skills_directory="./custom_skills", # Skills folder path
    security_level='moderate',          # strict|moderate|permissive
    auto_execute=True,                  # Auto-run generated code
    require_confirmation=False,         # Prompt before execution
    workspace_dir="./workspace",        # Output directory
    persistent=True,                    # Keep workspace between runs
    accumulate_blocks=True,             # Context between code blocks
    encoding='utf-8'                    # File encoding
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| execute_with_skill() | Execute task with skill context |
| execute_workflow() | Run multi-step workflow |
| list_skills() | List available skills |
| get_skill_info() | Get skill metadata |
| list_files() | List generated files |
| get_file() | Get file content |
| cleanup() | Clean workspace |

---

## Security Levels

| Level | Allows | Blocks | Use Case |
|-------|--------|--------|----------|
| strict | Standard lib | subprocess, eval, exec | Untrusted inputs |
| moderate | subprocess | eval, exec, compile | Recommended |
| permissive | Most ops | Minimal | Trusted only |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| ExpiredTokenException | Refresh AWS credentials: aws configure |
| No code blocks extracted | Add "Write Python code" to prompt |
| FileNotFoundError | Copy input files to workspace directory |
| Unicode errors | Set encoding='utf-8' in agent init |

### Debug Mode

```python
for i, result in enumerate(workflow_results, 1):
    print(f"Step {i}: Success={result['success']}, Files={result['files']}")
    for ex in result.get('executions', []):
        if ex.get('error'):
            print(f"  Error: {ex['error'][:200]}")
```

---

## Key Differences from Anthropic Skills API

| Feature | Anthropic API | This Implementation |
|---------|---------------|---------------------|
| Skill Upload | API-based | Local directory |
| Code Execution | Managed sandbox | Local subprocess |
| File Handling | API upload/download | Direct file access |
| Asset Loading | Automatic | Controlled via flags |
| Pricing | Anthropic rates | AWS Bedrock rates |

### Advantages of This Approach

- **Full Control**: Manage execution environment
- **AWS Integration**: Native Bedrock + IAM
- **Flexibility**: Granular asset loading
- **Cost Optimization**: Better for high-volume
- **Offline Skills**: No upload needed

---

## Estimated Costs

| Part | Input Tokens | Output Tokens | Est. Cost |
|------|--------------|---------------|-----------|
| Part 1 | ~15,000 | ~5,000 | ~0.15 USD |
| Part 2 | ~25,000 | ~10,000 | ~0.30 USD |
| Part 4 | ~40,000 | ~15,000 | ~0.50 USD |
| **Total** | **~80,000** | **~30,000** | **~0.95 USD** |

Costs vary by Claude model and AWS region.

---

## Additional Resources

- [Anthropic Skills Guide](https://platform.claude.com/docs/en/build-with-claude/skills-guide)
- [Anthropic Code Execution Tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)
- [Claude Cookbook: Skills](https://github.com/anthropics/claude-cookbooks/tree/main/skills)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

---

## Contributing

1. Fork the repository
2. Create feature branch: git checkout -b feature/amazing-feature
3. Commit changes: git commit -m 'Add amazing feature'
4. Push to branch: git push origin feature/amazing-feature
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Based on [Anthropic's Skills API](https://platform.claude.com/docs/en/build-with-claude/skills-guide)
- Inspired by [Deep Learning AI's Skills Course](https://github.com/https-deeplearning-ai/sc-agent-skills-files)
