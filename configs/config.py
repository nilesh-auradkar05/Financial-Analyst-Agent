"""
This file contains the configuration for the Financial Analyst Agent.
"""

from email.policy import default
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from pathlib import Path

""" PATHS CONFIGURATION """

# Get the absolute path of the config directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
FILINGS_DIR.mkdir(exist_ok=True)

""" SETTINGS CLASSES """


class OllamaSettings(BaseSettings):
    """
    Configuration for OLLAMA.

    Assuming that the Ollama server is running on the default port (11434)

    ENVIRONMENT VARIABLES:
           OLLAMA_BASE_URL: URL WHERE OLLAMA SERVER IS RUNNING
           OLLAMA_LLM_MODEL: MODEL TO USE FOR REASONING (eg: qwen3-vl 8B)
           OLLAMA_EMBED_MODEL: MODEL TO USE FOR EMBEDDINGS
           OLLAMA_TIMEOUT: REQUEST TIMEOUT IN SECONDS
    """

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base URL for Ollama server
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )

    # Main LLM model to use for reasoning tasks
    llm_model: str = Field(
        default="qwen3-vl:8b", description="Ollama model for reasoning"
    )

    embed_model: str = Field(
        default="nomic-embed-text", description="Ollama model for embeddings"
    )

    timeout: int = Field(default=600, description="Request timeout in seconds (10 minutes)")

    temperature: float = Field(default=0.7, description="LLm temperature(0.0 to 1.0)")


class TavilySettings(BaseSettings):
    """
    Configuration for Tavily Search API

    Environment Variables:
           TAVILY_API_KEY: API KEY FOR TAVILY SEARCH API
    """

    model_config = SettingsConfigDict(
        env_prefix="TAVILY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, description="Tavily API key")

    max_results: int = Field(
        default=5, description="Maximum number of search results per query"
    )

    search_depth: str = Field(
        default="advanced", description="Search depth (basic or advanced)"
    )


class LangSmithSettings(BaseSettings):
    """
    Configuration for LangSmith observability

    Environment Variables:
           LANGCHAIN_TRACING_V2: Enable/Disable LangSmith tracing(true or false)
           LANGCHAIN_API_KEY: API key for LangSmith
           LANGCHAIN_PROJECT: Project name for grouping traces
    """

    model_config = SettingsConfigDict(
        env_prefix="LANGCHAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tracing_v2: str = Field(
        default="true",
        alias="LANGCHAIN_TRACING_V2",
        description="Enable LangSmith tracing.",
    )

    api_key: Optional[str] = Field(default=None, description="LangSmith API Key")

    project: str = Field(
        default="financial-analyst-system", description="LangSmith project name"
    )


class ChromaDBSettings(BaseSettings):
    """
    Configuration for ChromaDB vector database.

    Environment Variables:
           CHROMA_PERSIST_DIR: Directory to store ChromaDB data
           CHROMA_COLLECTION_NAME: Name of the vector collection.
    """

    model_config = SettingsConfigDict(
        env_prefix="CHROMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Directory to store ChromaDB data
    # Using project's data directory by default
    persist_dir: str = Field(
        default=str(DATA_DIR / "chroma"), description="ChromaDB persistence directory"
    )

    # Collection name for SEC Filings
    collection_name: str = Field(
        default="sec_filings", description="ChromaDB collection name"
    )


class FinBERTSettings(BaseSettings):
    """
    Configuration for FinBERT sentiment analysis model

    Environment Variables:
           FINBERT_MODEL_NAME: HuggingFace model identifier
           DEVICE: Device to use for inference (cpu or cuda)
    """

    model_config = SettingsConfigDict(
        env_prefix="FINBERT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # HuggingFace model name
    model_name: str = Field(
        default="ProsusAI/finbert", description="FinBERT model name"
    )

    # Device to use for inference (cpu or cuda)
    device: Optional[str] = Field(
        default=None, description="Device to use for inference"
    )


class SECSettings(BaseSettings):
    """
    Configuration for SEC EDGAR API access.

    Environment Variables:
           SEC_USER_AGENT: User-Agent string with your email
    """

    model_config = SettingsConfigDict(
        env_prefix="SEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # User=Agent header(SEC requires this)
    user_agent: str = Field(
        default="FinancialAnalyst/1.0 nilesh.auradkar.0599@gmail.com",
        description="SEC EDGAR User-Agent",
    )

    base_url: str = Field(
        default="https://www.sec.gov", description="SEC EDGAR base URL"
    )

    # Rate Limit delay ( SEC asks for max 10 requests/second)
    rate_limit_delay: float = Field(
        default=0.1, description="Delay between SEC requests (seconds)"
    )


""" MAIN Settings Class"""


class Settings(BaseSettings):
    """
    Main Settings class that aggregates all configtings.s.

    Usage:
        from config import settings

        # Access nested settings
        model = settings.ollama.llm_model
        api_key = settings.tavily.api_key

    Environment:
        All settings can be overriden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested Settings objects
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    tavily: TavilySettings = Field(default_factory=TavilySettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    chroma: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    finbert: FinBERTSettings = Field(default_factory=FinBERTSettings)
    sec: SECSettings = Field(default_factory=SECSettings)

    # Application-level settings
    debug: bool = Field(default=True, description="Enable debug mode")

    log_level: str = Field(default="INFO", description="Logging level")


settings = Settings()


def validate_settings() -> list[str]:
    """
    Validate settings and return list of warning.

    Returns:
        List of warning messages for missing/invalid configuration.
    """
    warnings = []

    if not settings.tavily.api_key:
        warnings.append(
            "TAVILY_API_KEY not set. Web Search will not wark."
            "Get a free API key at https://tavily.com"
        )

    if not settings.langsmith.api_key:
        warnings.append(
            "LANGCHAIN_API_KEY not set. Tracing will not work."
            "Get a free API key at https://smith.langchain.com"
        )
    return warnings


""" MODULE EXPORTS """

__all__ = [
    "settings",
    "validate_settings",
    "PROJECT_ROOT",
    "DATA_DIR",
    "FILINGS_DIR",
]
