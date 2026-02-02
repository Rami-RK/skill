import json
import os
import boto3
import logging
import requests
from typing import Optional
import time, random, logging
from botocore.exceptions import ClientError
from botocore.config import Config

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaudeAPIS35:
    """Client for interacting with Claude AI API."""
    
    def __init__(self, api_endpoint):
        """
        Initialize Claude API client with API endpoint.
        
        Args:
            api_endpoint: Endpoint URL for Claude API.
        """
        self.api_endpoint = api_endpoint

    def invoke_llm_model(self, prompt):
        """
        Invoke Claude model with prompt.
        
        Args:
            prompt: Text prompt to send to Claude.
            
        Returns:
            Claude response as string.
        """
        try:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 20000,
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.post(self.api_endpoint, json=payload, headers=headers)
            
            if response.status_code == 200:
                claude_response = response.json()
                
                if 'content' in claude_response and isinstance(claude_response['content'], list):
                    return claude_response['content'][0]['text']
                elif 'completion' in claude_response:
                    return claude_response['completion']
                elif 'body' in claude_response:
                    body = json.loads(claude_response['body'])
                    if 'content' in body and isinstance(body['content'], list):
                        return body['content'][0]['text']
                    elif 'completion' in body:
                        return body['completion']
                else:
                    logger.error("Unable to find response text in the API response.")
                    return None
            else:
                logger.error(f"Error from Claude API: {response.status_code} - {response.text}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error in API request: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing API response: {e}")
            return None 


class ClaudeAPI:
    """
    Bedrock runtime client that invokes Claude via an inference profile ARN.
    Retries adaptively on throttling and respects Bedrock headers.
    """
    def __init__(self, region_name='eu-west-1', profile_arn=None):
        self.profile_arn = profile_arn or os.getenv("CLAUDE_INFERENCE_PROFILE_ARN")
        if not self.profile_arn:
            raise ValueError("Missing inference profile ARN (set CLAUDE_INFERENCE_PROFILE_ARN).")
        if "eu-west-1-1" in self.profile_arn:
            raise ValueError("Invalid region in ARN: use 'eu-west-1'.")

        # Use adaptive retry mode so the SDK backs off automatically as well
        cfg = Config(
            retries={"mode": "adaptive", "max_attempts": 15},
            read_timeout=120,
            connect_timeout=10,
        )
        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name, config=cfg)

    def _approx_tokens(self, s: str) -> int:
        # crude estimate: ~4 chars per token
        return max(1, len(s) // 4)

    def invoke_llm_model(self, prompt: str, max_tokens: int = 20000, temperature: float = 0.0) -> str:
        """
        Invoke Claude via Bedrock; handles throttling with exponential backoff + jitter.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }

        # Client-side soft guard: if the prompt is huge, shave it a bit
        # (keeps you under TPM limits more easily)
        if self._approx_tokens(prompt) + max_tokens > 180000:  # soft ceiling
            # keep last ~600k chars (~150k tokens)
            prompt = prompt[-600000:]
            body["messages"][0]["content"][0]["text"] = prompt

        throttling_codes = {
            "ThrottlingException", "Throttling", "TooManyRequestsException",
            "ProvisionedThroughputExceededException"
        }

        base = 0.8  # seconds
        max_sleep = 20.0
        attempts = 0
        max_attempts = 12

        while True:
            attempts += 1
            try:
                resp = self.bedrock.invoke_model(
                    modelId=self.profile_arn,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                payload = json.loads(resp["body"].read())

                # Anthropic content blocks → concatenate text parts
                parts = payload.get("content", [])
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
                return text or json.dumps(payload)

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in throttling_codes and attempts < max_attempts:
                    # Respect server hints if present
                    headers = e.response.get("ResponseMetadata", {}).get("HTTPHeaders", {}) or {}
                    retry_after = headers.get("retry-after")
                    if retry_after:
                        try:
                            sleep_s = float(retry_after)
                        except ValueError:
                            sleep_s = min(max_sleep, base * (2 ** (attempts - 1)))
                    else:
                        sleep_s = min(max_sleep, base * (2 ** (attempts - 1)))
                    # Add jitter
                    sleep_s += random.uniform(0, 0.5)
                    logger.warning("Throttled by Bedrock (%s). Backing off for %.2fs (attempt %d/%d).",
                                   code, sleep_s, attempts, max_attempts)
                    time.sleep(sleep_s)
                    continue
                # Not throttling or out of attempts → bubble up
                raise Exception(f"Error invoking Claude model: {e}") from e
            except Exception as e:
                # Non-ClientError exceptions (network, parse, etc.)
                if attempts < max_attempts and "Too many tokens" in str(e):
                    sleep_s = min(max_sleep, base * (2 ** (attempts - 1))) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
                    continue
                raise Exception(f"Error invoking Claude model: {e}") from e
