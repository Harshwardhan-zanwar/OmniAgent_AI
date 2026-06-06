"""
Conversation state management.
  - Primary  : LangChain ConversationBufferWindowMemory ->in-memory
  - Upgrade  : Redis-backed persistence (if UPSTASH_REDIS_URL is set)
  - Keeps last 5 exchanges per session
  - Auto-expires sessions after 24 hours (Redis) or server restart (memory)
"""

import os
import json
import logging
from collections import defaultdict

logger = logging.getLogger("omni-agent-ai.utils.state")

MEMORY_WINDOW = 5

REDIS_URL = os.getenv("UPSTASH_REDIS_URL", "")
_USE_REDIS = bool(REDIS_URL)

if _USE_REDIS:
    try:
        import redis as redis_lib
        _redis_client=redis_lib.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        logger.info("State backend: Redis (Upstash)")
    except Exception as exc:
        logger.warning(f"Redis connection failed:{exc} falling back to in-memory")
        _USE_REDIS=False

if not _USE_REDIS:
    logger.info("State backend: In-memory (LangChain ConversationBufferWindowMemory)")

try:
    from langchain.memory import ConversationBufferWindowMemory
    _memory_store:dict[str,ConversationBufferWindowMemory]={}

    def _get_lc_memory(session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in _memory_store:
            _memory_store[session_id]=ConversationBufferWindowMemory(
                k=MEMORY_WINDOW,
                return_messages=True,
                memory_key="history",)
        return _memory_store[session_id]
    _LANGCHAIN_AVAILABLE=True

except ImportError:
    _LANGCHAIN_AVAILABLE=False
    _plain_store:dict[str,list]=defaultdict(list)

def add_turn(session_id: str, query: str, result: str) -> None:
    """Save one user/assistant exchange to the session history."""
    if _USE_REDIS:
        _redis_add_turn(session_id,query,result)
    elif _LANGCHAIN_AVAILABLE:
        _lc_add_turn(session_id,query,result)
    else:
        _plain_add_turn(session_id,query,result)

def get_history(session_id: str) -> list[dict]:
    """
    Return conversation history as a list of dicts:
    [{"role":"user"|"assistant", "content":"..."}]
    """
    if _USE_REDIS:
        return _redis_get_history(session_id)
    elif _LANGCHAIN_AVAILABLE:
        return _lc_get_history(session_id)
    else:
        return _plain_get_history(session_id)

def get_history_as_gemini_format(session_id:str) -> list[dict]:
    """
    Return history formatted for Gemini's start_chat(history=...) API.
    [{"role":"user"|"model", "parts":["..."]}]
    """
    history=get_history(session_id)
    return[
        {
            "role":"model"if turn["role"]=="assistant" else"user",
            "parts":[turn["content"]],
        }
        for turn in history
    ]

def clear_session(session_id:str) -> None:
    """Delete all history for a session."""
    if _USE_REDIS:
        _redis_client.delete(f"session:{session_id}")
    elif _LANGCHAIN_AVAILABLE and session_id in _memory_store:
        del _memory_store[session_id]
    elif not _LANGCHAIN_AVAILABLE:
        _plain_store.pop(session_id,None)
    logger.info(f"Session cleared: {session_id}")


#lc backend
def _lc_add_turn(session_id:str,query:str,result:str) -> None:
    mem=_get_lc_memory(session_id)
    mem.save_context(
        {"input":query},
        {"output":result[:600]},
    )
    logger.debug(f"[LC] Saved turn for session {session_id}")

def _lc_get_history(session_id:str) -> list[dict]:
    mem=_get_lc_memory(session_id)
    messages=mem.load_memory_variables({}).get("history",[])
    history=[]
    for msg in messages:
        role="user" if msg.__class__.__name__=="HumanMessage" else "assistant"
        history.append({"role":role,"content":msg.content})
    return history


# redis backend
def _redis_add_turn(session_id:str,query:str,result:str) -> None:
    key=f"session:{session_id}"
    history=_redis_get_history(session_id)
    history.extend([
        {"role":"user",      "content": query},
        {"role":"assistant", "content": result[:600]},
    ])
    history = history[-(MEMORY_WINDOW * 2):]
    _redis_client.setex(key, 86400, json.dumps(history))  # 24hr TTL
    logger.debug(f"[Redis] Saved turn for session {session_id}")

def _redis_get_history(session_id: str) -> list[dict]:
    key =f"session:{session_id}"
    data=_redis_client.get(key)
    return json.loads(data) if data else []


# plain python
def _plain_add_turn(session_id:str,query:str,result:str) -> None:
    _plain_store[session_id].extend([
        {"role":"user","content":query},
        {"role":"assistant","content":result[:600]},
    ])
    _plain_store[session_id]=_plain_store[session_id][-(MEMORY_WINDOW*2):]


def _plain_get_history(session_id:str) -> list[dict]:
    return list(_plain_store.get(session_id,[]))