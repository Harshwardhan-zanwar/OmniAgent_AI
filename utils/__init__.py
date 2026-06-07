"""
Usage anywhere in the project:
    from utils import add_turn, get_history         
    from utils import get_cost                  
    from utils import setup_logging, format_trace 
"""

from utils.state import (
    add_turn,
    get_history,
    get_history_as_gemini_format,
    clear_session,
)

from utils.cost_estimator import (
    get_cost,
    CostEstimate,
)

from utils.logger import (
    setup_logging,
    format_trace,
    format_trace_json,
    RequestLogger,
)

__all__ = [
    # state
    "add_turn",
    "get_history",
    "get_history_as_gemini_format",
    "clear_session",
    # cost
    "get_cost",
    "CostEstimate",
    # logger
    "setup_logging",
    "format_trace",
    "format_trace_json",
    "RequestLogger",
]