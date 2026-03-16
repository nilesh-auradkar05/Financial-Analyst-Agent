"""
Financial Analyst Agents System
===============================

Modules:
    graph: Original LangGraph Single-agent Workflow
    crew: CrewAI Multi-Agent system (4 Specialized Agents)
    state: Shared State Definitions
"""

from agents.crew_agents import CREWAI_AVAILABLE
from agents.graph import create_agent, run_agent, run_agent_sync
from agents.state import AgentState, AgentStep, create_initial_state

# CrewAI (Optional - graceful import)
try:
    from agents.crew_agents import CrewResult, run_crew_analysis, run_crew_sync
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    run_crew_analysis = None
    run_crew_sync = None
    CrewResult = None

__all__ = [
    # LangGraph
    "AgentState",
    "AgentStep",
    "create_initial_state",
    "run_agent",
    "run_agent_sync",
    "create_agent",
    # CrewAI
    "run_crew_analysis",
    "run_crew_sync",
    "CrewResult",
    "CREWAI_AVAILABLE",
]
