import os

os.environ["LLM_PROVIDER"] = "bedrock"
os.environ["LLM_MODEL"] = "us.anthropic.claude-sonnet-4-6"
os.environ["AWS_REGION"] = "us-east-1"

from app.config import Settings
from app.llm.provider import get_llm

llm = get_llm(Settings())
print(llm.invoke("Reply with single word: ok").content)