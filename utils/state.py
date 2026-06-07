"""Conversation state management."""
import os
import json
import logging
from collections import defaultdict

logger=logging.getLogger("omni-agent-ai.utils.state")
window=5

redis_url=os.getenv("UPSTASH_REDIS_URL","")
use_redis=bool(redis_url)

if use_redis:
    try:
        import redis as redis_lib
        redis_client=redis_lib.from_url(redis_url,decode_responses=True)
        redis_client.ping()
        logger.info("State backend: Redis")
    except Exception as e:
        logger.warning(f"Redis failed: {e} — using memory")
        use_redis=False

try:
    from langchain.memory import ConversationBufferWindowMemory
    mem_store={}

    def _get_lc_memory(sid:str) -> ConversationBufferWindowMemory:
        if sid not in mem_store:
            mem_store[sid]=ConversationBufferWindowMemory(
                k=window,
                return_messages=True,
                memory_key="history",
            )
        return mem_store[sid]
    langchain_ok=True
except ImportError:
    langchain_ok=False
    plain_store=defaultdict(list)

def add_turn(session_id:str,query:str,result:str) -> None:
    if use_redis:
        _redis_add_turn(session_id,query,result)
    elif langchain_ok:
        _lc_add_turn(session_id,query,result)
    else:
        _plain_add_turn(session_id,query,result)

def get_history(sid:str) -> list[dict]:
    if use_redis:
        return _redis_get_history(sid)
    elif langchain_ok:
        return _lc_get_history(sid)
    return _plain_get_history(sid)

def get_history_as_gemini_format(sid:str) -> list[dict]:
    history=get_history(sid)
    return [
        {
            "role":"model" if turn["role"]=="assistant" else "user",
            "parts":[{"text": turn["content"]}],
        }
        for turn in history
    ]

def clear_session(sid:str) -> None:
    if use_redis:
        redis_client.delete(f"session:{sid}")
    elif langchain_ok and sid in mem_store:
        del mem_store[sid]
    elif not langchain_ok:
        plain_store.pop(sid,None)
    logger.info(f"Cleared session: {sid}")

def _lc_add_turn(sid:str,q:str,res:str) -> None:
    mem=_get_lc_memory(sid)
    mem.save_context({"input":q},{"output":res[:600]})

def _lc_get_history(sid:str) -> list[dict]:
    mem=_get_lc_memory(sid)
    msgs=mem.load_memory_variables({}).get("history",[])
    history=[]
    for msg in msgs:
        role="user" if msg.__class__.__name__=="HumanMessage" else "assistant"
        history.append({"role":role,"content":msg.content})
    return history

def _redis_add_turn(sid:str,q:str,res:str) -> None:
    key=f"session:{sid}"
    history=_redis_get_history(sid)
    history.extend([
        {"role":"user","content":q},
        {"role":"assistant","content":res[:600]},
    ])
    history=history[-(window*2):]
    redis_client.setex(key,86400,json.dumps(history))

def _redis_get_history(sid:str) -> list[dict]:
    key=f"session:{sid}"
    data=redis_client.get(key)
    return json.loads(data) if data else []

def _plain_add_turn(sid:str,q:str,res:str) -> None:
    plain_store[sid].extend([
        {"role":"user","content":q},
        {"role":"assistant","content":res[:600]},
    ])
    plain_store[sid]=plain_store[sid][-(window*2):]

def _plain_get_history(sid:str) -> list[dict]:
    return list(plain_store.get(sid,[]))