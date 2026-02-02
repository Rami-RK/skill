
import os
from dotenv import load_dotenv

from claude_client import ClaudeAPIS35, ClaudeAPI

load_dotenv()

PROMPT = "list 5 facts about Taj Mahal"

CLAUDE_45_INFERENCE_PROFILE_ARN = os.getenv("CLAUDE_45_INFERENCE_PROFILE_ARN")


def call_claude_35() -> None:
    endpoint = os.getenv("CLAUDE_API_ENDPOINT")
    if not endpoint:
        print("Missing CLAUDE_API_ENDPOINT; skipping claude-3.5-sonnet.")
        return
    client = ClaudeAPIS35(endpoint)
    output = client.invoke_llm_model(PROMPT)
    print("claude-3.5-sonnet output:")
    print(output)


def call_claude_bedrock(profile_arn: str) -> None:
    if not profile_arn:
        print("Missing CLAUDE_INFERENCE_PROFILE_ARN; skipping Bedrock Claude.")
        return
    region = os.getenv("AWS_REGION", "eu-west-1")
    client = ClaudeAPI(region_name=region, profile_arn=profile_arn)
    output = client.invoke_llm_model(PROMPT, max_tokens=256, temperature=0.2)
    print("claude-3.7/4.x via Bedrock output:")
    print(output)


if __name__ == "__main__":
    #call_claude_35()
    call_claude_bedrock(CLAUDE_45_INFERENCE_PROFILE_ARN)
