import os

os.environ["LLM_PROVIDER"] = "bedrock"
os.environ["LLM_MODEL"] = "deepseek.v3.2"
os.environ["LLM_THINKING_MODE"] = "enabled"
os.environ["AWS_REGION"] = "us-east-1"
from app.config import Settings
from app.llm.provider import get_llm

r = get_llm(Settings()).invoke("In one sentence: is 9.11 or 9.8 larger?")
print(r.content)
print("REASONING?", r.additional_kwargs.get("reasoning_content") or r.response_metadata)