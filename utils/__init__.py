"""
Usage anywhere in the project:
    from utils import add_turn, get_history         
    from utils import estimate_cost                  
    from utils import setup_logging, format_plan_trace 
"""

from utils.state import (
    add_turn,
    get_history,
    get_history_as_gemini_format,
    clear_session,
)

from utils.cost_estimator import (
    estimate_cost,
    CostEstimate,
)

from utils.logger import (
    setup_logging,
    format_plan_trace,
    format_plan_trace_json,
    RequestLogger,
)

__all__ = [
    # state
    "add_turn",
    "get_history",
    "get_history_as_gemini_format",
    "clear_session",
    # cost
    "estimate_cost",
    "CostEstimate",
    # logger
    "setup_logging",
    "format_plan_trace",
    "format_plan_trace_json",
    "RequestLogger",
]