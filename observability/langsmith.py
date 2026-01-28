"""
FINANCIAL ANALYST AGENT SYSTEM - LANGSMITH INTEGRATION

"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain_core.tracers import LangChainTracer as LangChainTracerType
    from langsmith import Client as ClientType

from loguru import logger

try:
    from langchain_core.tracers import LangChainTracer
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    Client = None # type: ignore
    LangChainTracer = None # type: ignore
    LANGSMITH_AVAILABLE = False

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings

# =============================================================================
# SETUP
# =============================================================================


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return bool(
        LANGSMITH_AVAILABLE
        and settings.langsmith.tracing_v2
        and bool(settings.langsmith.api_key)
    )


def setup_langsmith_env() -> bool:
    """
    Set up environment variables for LangSmith.

    Call this at application startup.
    Returns: True if tracing is enabled.
    """
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not installed - tracing disabled")
        return False

    if not settings.langsmith.api_key:
        logger.warning("LANGCHAIN_API_KEY not set - tracing disabled")
        return False

    # LangSmith reads these env vars automatically
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith.tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith.api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project

    logger.info(f"LangSmith tracing enabled (project: {settings.langsmith.project})")
    return True


# =============================================================================
# TRACER
# =============================================================================


def get_tracer() -> Optional["LangChainTracerType"]:
    """
    Get tracer for LangGraph integration.

    Usage:
        tracer = get_tracer()
        config = {"callbacks": [tracer]} if tracer else {}
        result = await agent.ainvoke(state, config=config)
    """
    if not is_tracing_enabled() or LangChainTracer is None:
        return None

    try:
        return LangChainTracer(project_name=settings.langsmith.project)
    except Exception as e:
        logger.error(f"Failed to create tracer: {e}")
        return None


def get_client() -> Optional["ClientType"]:
    """Get LangSmith client for advanced operations."""
    if not is_tracing_enabled() or Client is None:
        return None

    try:
        return Client()
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        return None


# =============================================================================
# HEALTH CHECK
# =============================================================================


async def check_langsmith_connection() -> dict:
    """Check LangSmith connection status."""
    result = {
        "available": LANGSMITH_AVAILABLE,
        "enabled": is_tracing_enabled(),
        "project": settings.langsmith.project,
        "connected": False,
    }

    if not is_tracing_enabled():
        return result

    try:
        client = get_client()
        if client:
            list(client.list_projects(limit=1))
            result["connected"] = True
    except Exception as e:
        result["error"] = str(e)

    return result
