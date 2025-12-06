"""
This module provides integration with ollama for local LLM inference.

I am using Qwen3-VL model which can:
    - Process text prompts
    - Analyze images (in this context it is useful to process charts, tables, diagrams from SEC filings).

Usage:
------------------------
    from models.llm import get_llm, analyze_image, generate_text

    # langchain-compatible llm
    llm = get_llm()
    response = llm.invoke("Explain P/E ration")

    # direct text generation
    response = await generate_text("Summarize these risk factors: .......")

    # Vision analysis (for chart/tables)
    analysis = await analyze_image(
        image_path = "chart.png",
        prompt="Describe the revenue trend shown in this chart."
    )

"""

import asyncio
import base64
import httpx
from pathlib import Path
from typing import Optional, Union, AsyncGenerator
from dataclasses import dataclass
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings

# DATA Models

@dataclass
class LLMResponse:
    """
    Standardize response from LLM calls.

    Wraps both Langchain and direct API responses in a consistent format.

    Attributes:
        - content: The generated text response
        - model: Model that generated the response
        - prompt_tokens: Number of tokens in the prompt
        - completion_tokens: Number of tokens in the response
        - total_duration_ms: Total generation time in milliseconds
        - error: Error message if generation failed
    """

    content: str
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_duration_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Was generation successfull"""
        return self.error is None and len(self.content) > 0

    @property
    def total_tokens(self) -> Optional[int]:
        """Total tokens used (prompt + completion)"""
        if self.prompt_tokens and self.completion_tokens:
            return self.prompt_tokens + self.completion_tokens

        return None

@dataclass
class VisionResponse(LLMResponse):
    """
    Response from vision-language model analysis.

    Attributes:
        image_path: Path to the analyzed image
        image_size_bytes: Size of the image in bytes
    """

    image_path: Optional[str] = None
    image_size_bytes: Optional[int] = None

# Langchain Integration

def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> ChatOllama:
    """
    Get a Langchain-compatible ChatOllama Instance.

    Args:
        model: Ollama model name. Defaults to settings.ollama.llm_model
        temperature: Generation temperature (0.0 to 1.0). The lower the temperature more deterministic is the output of the model.
        base_url: Ollama server URL. Defaults to settings.ollama.base_url
        timeout: Request timeout in seconds

    Returns:
        ChatOllama instance

    Example:
        llm = get_llm()

        # initialization
        response = llm.invoke("What is a P/E ration?")
        print(response.content)

        # invoke with message history
        messages = [
            SystemMessage(content="You are a financial analyst."),
            HumanMessage(content="Analyze Apple's risk factors"),
        ]

        response = llm.invoke(messages)
    """

    # instantiate with values or fall back to default settings
    model = model or settings.ollama.llm_model
    temperature = temperature if temperature is not None else settings.ollama.temperature
    base_url = base_url or settings.ollama.base_url
    timeout = timeout or settings.ollama.timeout

    logger.info(f"Creating ChatOllama (model={model}, temperature={temperature})")

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        # Ollama Specific settings
        num_ctx = 8192, # context window size
        num_predict=-1, # No limit on output tokens (-1 = unlimited)
    )

def get_llm_with_system_prompt(
    system_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOllama:
    """
    Get an LLM with a bound system prompt. (Standard practice)

    This helps to get consistent behaviour from the llm across mulltiple calls.

    Args:
        system_prompt: System message to prepend to all conversations
        model: Ollama model name
        temperature: Generation temperature

    Returns:
        ChatOllama instance with bound system prompt.

    Example:
        analyst_llm = get_llm_with_system_prompt(
                        "You are a senior financial analyst at a hedge fund. Provide detailed, data-driven analysis."
                    )
        response = analyst_llm.invoke("Analyze AAPL's recent performance")
    """

    llm = get_llm(model=model, temperature=temperature)

    return llm.bind(system=system_prompt)

# Direct OLLAMA API Client

class OllamaClient:
    """
    Ollama Rest API client.

    The primary functionality of this is to:
        - multimodel tasks/ vision tasks
        - Streaming responses
        - Fine-grained control over generation parameters
    
    Example:
        async with OllamaClient() as client:
            # Text generation
            response = await client.generate("Explain market cap")

            # Vision analysis
            response = await client.generate_with_image(
                prompt="Describe this chart",
                image_path="revenue_chart.png",
            )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[str] = None,
    ):
        """
        Inititalize Ollama Client.

        Args:
            base_url: Ollama server URL
            model: Default model to use
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self.model = model or settings.ollama.llm_model
        self.timeout = timeout or settings.ollama.timeout

        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"OllamaClient initialized (base_url={self.base_url}, model={self.model})")

    async def __aenter__(self):
        """Async context manager entry."""
        # Use longer read timeout for LLM generation (can take several minutes)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=30.0,        # 30s to establish connection
                read=600.0,          # 10 minutes for LLM to generate response
                write=30.0,          # 30s to send request
                pool=30.0,           # 30s to acquire connection from pool
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with OllamaClient() as client'"
            )
        return self._client

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64.

        Args:
            imaage_path: Path to image file

        Retirms:
            Base64-encoded string
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:

        """
        Generate text from a prompt.

        Args:
            prompt: User prompt/question
            model: Model to use
            system: System prompt
            temperature: Generation temperature
            max_tokens: Maximum tokes to generate

        Returns:
            LLMResponse with generated text
        """

        model = model or self.model
        temperature = temperature if temperature is not None else settings.ollama.temperature

        logger.debug(f"Generating with model={model}, prompt_len={len(prompt)}")

        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()

            data = response.json()

            return LLMResponse(
                content=data.get("response", ""),
                model=model,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_duration_ms=data.get("total_duration", 0) / 1_000_000      # converting nano-seconds to micro-seconds
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return LLMResponse(
                content="",
                model=model,
                error=str(e),
            )

    async def generate_with_image(
        self,
        prompt: str,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> VisionResponse:
        """
        Generate text from a prompt + image (vision tasks).

        Args:
            prompt: Question/instruction about the image
            image_path: Path to image file (PNG, JPG, etc.)
            model: Model to use
            system: System prompt
            temperature: Generation temperature

        Returns:
            VisionResponse with analysis

        Example:
            response = await client.generate_with_image(
                            prompt="What revenue trend does this chart show?",
                            image_path="sec_filing_chart.png"
                        )
        """
        model = model or self.model
        temperature = temperature if temperature is not None else settings.ollama.temperature
        image_path = Path(image_path)

        logger.info(f"Vision analysis: {image_path.name}")

        # Encode image
        try:
            image_b64 = self._encode_image(image_path=image_path)
            image_size = image_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return VisionResponse(
                content="",
                model=model,
                error=f"Failed to load image: {e}",
                image_path=str(image_path),
            )

        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temprature": temperature,
            },
        }

        if system:
            payload["system"] = system

        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()

            data = response.json()

            return VisionResponse(
                content=data.get("response", ""),
                model=model,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_cound"),
                total_duration_ms=data.get("total_duration", 0) / 1_000_000,
                image_path=str(image_path),
                image_size_bytes=image_size,
            )

        except Exception as e:
            logger.error(f"Vision generation failed: {e}")
            return VisionResponse(
                content="",
                model=model,
                error=str(e),
                image_path=str(image_path),
            )

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated text token by token.

        Args:
            prompt: User prompt
            model: Model to use
            system: System prompt
            temperature: Generartion temperature
        
        Yields:
            Text chunks as they're generated.

        Example:
            async for chunk in client.generate_stream("What is the financial trend of AAPL in last 5 years"):
                print(chunk, end="", flush=True)
        """
        model = model or self.model
        temperature = temperature if temperature is not None else settings.ollama.temperature

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield f"\n[Error: {e}]"

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        f"""
        Multi-turn chat completion

        Args:
            messages: List of message dicts with 'role' and 'content'
                        Roles: 'system', 'user', 'assistant'
            model: Model to use
            temperature: Generation temperature

        Returns:
            LLMResponse with assistant's reply

        Example:
            messages = [
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": "What is Apple's P/E ratio?"},
                {"role": "assistant", "content": "Apple's P/E ration is around 28."},
                {"role", "user", "content": "How does that compare to the industry?"},
            ]
            response = await client.chat(messages)
        """
        model = model or self.model
        temperature = temperature if temperature is not None else settings.ollama.temperature

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()

            data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=model,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_duration_ms=data.get("total_duration", 0) / 1_000_000,
            )

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return LLMResponse(
                content="",
                model=model,
                error=str(e),
            )

    async def health_check(self) -> bool:
        """
        Check if ollama server is running and model is available

        Returns:
            True if server is responsive and model loaded
        """
        try:
            # check server
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            
            # Check if model if available
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            if self.model in models or any(self.model in m for m in models):
                logger.info(f"Health check passed: {self.model} available")
                return True
            
            logger.warning(f"Model {self.model} not found. Available: {models}")
            return False

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Wrapper Convenience functions
# Module level functions

async def generate_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[int] = None,
) -> LLMResponse:

    """
    Generate text from a prompt.

    Args:
        prompt: user prompt
        system: Optional system prompt
        temperature: Generation temperature

    Returns:
        LLMResponse with generated text

    Example:
        response = await generate_text(
            "Summarize Apple's risk factors",
            system="You are a financial analyst."
        )
        print(response.content)
    """
    async with OllamaClient() as client:
        return await client.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
        )

async def analyze_image(
    image_path: Union[str, Path],
    prompt: str,
    system: Optional[str] = None,
) -> VisionResponse:

    """
    Analyze an image with the vision-language model.

    Args:
        image_path: Path to image file
        prompt: Question about the image
        system: Optional system prompt

    Returns:
        VisionResponse with analysis

    Example:
        response: await analyze_image(
            "sec_chart.png",
            "What trend does this revenue chart show?"
        )
        print(response.content)
    """

    async with OllamaClient() as client:
        return await client.generate_with_image(
            prompt=prompt,
            image_path=image_path,
            system=system,
        )

async def check_ollama_health() -> bool:
    """
    Check if Ollama is running and model is available.

    Returns:
        True if healthy, False otherwise
    """
    async with OllamaClient() as client:
        return await client.health_check()

# Financial Anaysis System Prompt

class FinancialPrompts:
    """
    Collection of prompts optimized for financial analysis tasks.

    I have curated 2 specialized system prompt for Analyst system and Risk analyst system.
    """

    # System Propmt for Financial analyst persona
    ANALYST_SYSTEM = """You are a senior financial analyst at a top-tier investment firm.
    Your analysis is thorough, data-driven, and objective. You:
    - Cite specific data points and sources
    - Acknowledge uncertainty and risks
    - Avoid Speculation without evidence
    - User precise financial terminology
    - Structure your analysis clearly
    """

    # Risk Analyst Prompt.
    RISKT_ANALYSIS = """Analyze the following Risk Factors section from a 10-K filing.
    
    Identify and categorize the key risks into:
    1. Market/Economic Risks.
    2. Operational Risks.
    3. Financial Risks.
    4. Regulatory/Legal Risks.
    5. Strategic Risks

    For each category, list the top 2-3 risks with a brief explanation of potential impact.

    Risk Factors:
    {content} 
    
    Provide your analysis in a structured format.
    """

    # Prompt for sentiment analysis context
    SENTIMENT_CONTEXT = """Based on the following news articles about {company}, assess the overall market sentiment.

    Articles:
    {articles}

    Provide:
        1. Overall Sentiment: (Bullish/Bearish/Neutral)
        2. Key Positive Factors
        3. Key Negative Factors
        4. Confidence Level: (High/Medium/Low)
        5. Brief Rationale (2-3 sentences)
    """

    # Prompt for investment memo summary
    MEMO_SUMMARY = """Gemerate an executive summary for an investment memo on {company} ({ticker}).

    Company Overview:
    {company_info}

    Recent Stock Performance:
    {stock_data}

    Key Risks (from 10-K):
    {risk_summary}

    News Sentiment:
    {sentiment}

    Write a 3-4 paragraph executive summary that:
        1. Summarizes the investment thesis
        2. Highlights key opportunities
        3. Acknowledge primary risks
        4. Provides a balanced recommendation

    Use a professional, Objective tone.    
    """

    # Prompt for chart/table analysis (vision)
    CHART_ANALYSIS = """Analyze this financial chart or table from a SEC filing.

    Describe:
        1. What type of visualization is this? (line chart, bar chart, table, etc.)
        2. What metrics or data does it show?
        3. What is the time period covered?
        4. What are the key trends or insights?
        5. Are there any notable anomalies or points of interest?

        Provide specific numbers where visible.
    """

# Testing
async def _main():
    """Test the LLM Module"""
    import sys

    print(f"\n{'='*60}")
    print("LLM Module - Testing")
    print(f"{'='*60}")

    # Test Ollama API
    print("1. Health check....")
    async with OllamaClient() as client:
        healthy = await client.health_check()
        if healthy:
            print(f"Ollama API is responsive and model is available\n")
        else:
            print("Ollama health check failed.")
            print("Make sure Ollama is running: Ollama serve")
            print(f"And model is pulled: Ollama pull {settings.ollama.llm_model}")

    # Test text generation
    print("2. Text Generation....")
    response = await generate_text(
        prompt="What is a P/E ratio? Explain in 2 sentences.",
        system="You are a helpful financial educator.",
    )

    if response.success:
        print(f"Generated response:")
        print(f'     {response.content[:200]}....\n')
        print(f"      Tokens: {response.total_tokens}, Time: {response.total_duration_ms}ms\n")
    else:
        print(f"Generation failed: {response.error}\n")

    # Test Langchain integration
    print("3. Langchain integration....")
    llm = get_llm()
    try:
        lc_response = llm.invoke("What is market capitalization? One sentence.")
        print(f"    Langchain reponse:")
        print(f"    {lc_response.content[:200]}...\n")
    except Exception as e:
        print(f"    Langchain failed: {e}\n")

    # Test streaming
    print("4. Streaming Generation....")
    print("   Response: ", end="")
    async with OllamaClient() as client:
        async for chunk in client.generate_stream(
            "What is Trading in financial market?",
        ):
            print(chunk, end="", flush=True)
    print("\n     Streaming complete\n")

    # Test vision(if image provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"5. Vision Analysis ({image_path})....")


        response = await analyze_image(
            image_path=image_path,
            prompt="Describe what you see in this image. Be Specific.",
        )

        if response.success:
            print(f"    Vision response:")
            print(f"    {response.content[:300]}.....")
        else:
            print(f"    Vision failed: {response.error}")
    else:
        print(f"5. Vision Analysis...")
        print("    Skipped (no image provided)")
        print("    Run with: python models/llm.py <image_path>")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"="*60)

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    asyncio.run(_main())
