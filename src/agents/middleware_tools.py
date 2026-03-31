"""Small helper utilities for capturing tool responses across agents.

The goal is to collect Azure Search payloads in one place for debugging
without forcing each agent to wire up custom middleware.
"""

import json
from typing import Any
from collections.abc import Awaitable, Callable
from loguru import logger
import time

from agent_framework import (
    Agent,
    AgentResponse,
    FunctionInvocationContext,  # Context passed to middleware during function/tool calls 
    FunctionMiddleware,  # Base class for creating function call middleware
    AgentMiddleware,  # Base class for agent-level middleware that hooks into agent lifecycle events
    AgentContext,  # Context passed to middleware during agent runs that includes run-specific data
)

# Global list to store serialized search evidence across agents (retriever, fact-checker)
search_evidence: list[str] = []

# Global list to store the actual search queries the agent passed to its tools
search_queries: list[str] = []

# Define which tools' results we want to capture as evidence for fact-checker agent
_SEARCH_TOOLS: set[str] = {"search_customers", "_search_customers",
                           "search_products", "_search_products"}

# Clear evidence between workflow runs to avoid data contamination:
def clear_evidence() -> None:
    """Reset every evidence bucket (collection of related evidence) 
    so each workflow run starts clean, without old data."""
    search_evidence.clear()
    search_queries.clear()


def _to_loggable(value: Any) -> Any:
    """Recursively convert framework objects into JSON-safe debug payloads."""

    if hasattr(value, "to_dict"):
        try:
            return _to_loggable(value.to_dict())
        except Exception:
            return repr(value)

    if hasattr(value, "model_dump"):
        try:
            return _to_loggable(value.model_dump())
        except Exception:
            return repr(value)

    if isinstance(value, dict):
        return {str(key): _to_loggable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_loggable(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def _record_search_payload(tool_name: str, arguments: Any, payload: Any) -> None:
    """This helper function serializes (stringifies) Azure Search hits 
    (search results), so that retriever & fact-checker agents can reuse them.
    It avoids making the LLM re-interpret tool results that can lead to hallucinations.
    This func is called by the middleware after each tool execution to capture results.
    
    Args:
        tool_name: Name of the tool that produced the payload
        arguments: Arguments (incl. query) passed to the tool by AI (BaseModel or dict)
        payload: The actual data returned by the tool (could be dict, list, etc.)
    """
    if tool_name not in _SEARCH_TOOLS or not payload:  # Ignore irrelevant tools or empty
        return  # Early exit if tool is not in our capture list

    # Capture the query parameter that was passed to the search tool
    # Convert BaseModel to dict if needed: check to see if arguments is BaseModel or dict
    args_dict = arguments.model_dump() if hasattr(arguments, 'model_dump') else arguments
    query = args_dict.get("query", "") if isinstance(args_dict, dict) else ""
    
    if query:
        search_queries.append(query)
        logger.debug(
            "[_record_search_payload] Captured search query for '{}': {}",
            tool_name, query
        )

    docs = payload if isinstance(payload, list) else [payload]

    for doc in docs:
        try:
            search_evidence.append(
                json.dumps(_to_loggable(doc), ensure_ascii=False)
            )
        except TypeError as exc:  # Rough logging so we can spot failures fast.
            logger.warning("Skipping evidence serialization for '{}': {}",
                           tool_name, exc)


# Helper to attach the middleware to multiple agents at once:
def attach_middlewares(*agents: Agent) -> None:
    """Add the capture middleware to each agent once (idempotent helper).
    It checks if the middleware is already present in each agent to avoid duplicates.
    
    Args:
        agents: Variable length argument list of agents to attach middleware to.
        `*` means you can pass any number of agents as separate arguments.
        e.g. attach_middlewares(agent1, agent2, ...) is valid.
    """
    for agent in agents:  # Iterate over each provided agent
        if agent is None:
            continue

        # making a list of current middleware attached to the agent.
        # `agent.middleware` may be None/tuple/list, so we normalize it to a list.
        current = list(getattr(agent, "middleware") or [])

        # If any instance of middlewares is present in the list, we skip adding it again:
        for midw_class in (ToolCaptureMiddleware, AgentCaptureMiddleware):
            if not any(isinstance(midw, midw_class) for midw in current):
                current.append(midw_class())

        agent.middleware = current


class ToolCaptureMiddleware(FunctionMiddleware):
    """Middleware that logs tool calls and displays/stores raw doc results."""

    # Middleware method that runs around each tool/function call.
    async def process(
        self,
        context: FunctionInvocationContext,  # tool call context (name, args, result etc.)
        call_next: Callable[[],
                            Awaitable[None]],  # The next middleware/tool in the chain
    ) -> None:
        """Middleware that runs at the tool/function level to log tool calls & capture results"""

        # Name supplied when the tool was registered
        tool_name = context.function.name
        start_time = time.perf_counter()

        logger.opt(colors=True).info("<yellow>[ToolCaptureMiddleware]</yellow> Tool invocation "
                    "started! Calling the function: '{}'",
                    tool_name)
        
        await call_next()  # Let the underlying tool run normally first.
        
        # Calculate execution time
        duration_ms = (time.perf_counter() - start_time) * 1000
              
        # Log tool execution
        logger.opt(colors=True).info(
            "<yellow>[ToolCaptureMiddleware]</yellow> Tool execution complete "
            "for function '{}' | Duration: {}ms",
            tool_name, duration_ms)

        # NOW, get the function result after the tool has executed
        tool_result = json.dumps(
            _to_loggable(context.result if context.result else {}),
            indent=3,
            ensure_ascii=False,
        )

        tool_arguments = context.arguments # Get the tool's input arguments

        logger.opt(colors=True).debug("<yellow>[ToolCaptureMiddleware] '{}' "
                                       "function's args:\n{}</yellow>",
                                       tool_name, tool_arguments)

        logger.opt(colors=True).debug("<yellow>[ToolCaptureMiddleware] '{}' "
                                       "function's captured result(s):\n{}</yellow>",
                                       tool_name, tool_result)
        
        # Record the ai search payloads for fact-checker grounding checks,
        # if applicable, meaning that the tool has to be in our predefined list.
        # Pass both arguments (containing query) and results
        _record_search_payload(tool_name, context.arguments, context.result)


class AgentCaptureMiddleware(AgentMiddleware):
    """Agent-level middleware that logs agent lifecycle events."""
    
    # Middleware method that runs around each agent run.
    async def process(
        self,
        # context passed in agent midware pipeline w all info about agent invocation:
        context: AgentContext,
        call_next: Callable[[],
                            Awaitable[None]],  # The next middleware/agent in the chain
    ) -> None:
            
        agent_name = context.agent.name
        start_time = time.perf_counter()

        """Middleware that runs at the agent level to log agent lifecycle events."""
        logger.opt(colors=True).info("<magenta>[AgentCaptureMiddleware]</magenta> Agent-level "
                    "capture middleware starting execution for agent: '{}'",
                    agent_name)

        await call_next()  # Let the agent run normally first.

        duration_ms = (time.perf_counter() - start_time) * 1000 

        logger.opt(colors=True).info(
            "<magenta>[AgentCaptureMiddleware]</magenta> Agent execution "
            "complete for agent '{}' | Duration: {}ms",
            agent_name, duration_ms)

        # Safely extract the agent's final result
        raw_result = context.result

        # If there's no result at all, log an error and raise
        if raw_result is None:
            logger.opt(colors=True).error(
                "<magenta>[AgentCaptureMiddleware]</magenta> Agent '{}' "
                "finished with no result! Exiting logging.",
                agent_name)
            raise ValueError("Agent finished with no results!")

        # Serialize: AgentResponse has to_dict(); structured-output Pydantic models have model_dump()
        if isinstance(raw_result, AgentResponse):
            ctx_result = raw_result
            agent_result = json.dumps(
                ctx_result.to_dict(),
                indent=3,
                ensure_ascii=False,
            )
        elif hasattr(raw_result, "model_dump"):
            agent_result = json.dumps(
                _to_loggable(raw_result.model_dump()),
                indent=3,
                ensure_ascii=False,
            )
        else:
            agent_result = json.dumps(_to_loggable(raw_result), indent=3, ensure_ascii=False)

        # Serialize agent's context.messages: extract role & text from each ChatMessage
        agent_messages_list = [
            {
                "role": str(msg.role),
                "text": msg.text,
            } 
            for msg in context.messages
        ]

        # # VERY VERY verbose logging of all agent messages, not just role & text
        # agent_messages_list = [
        #     msg.to_dict() for msg in context.messages
        # ]

        # Convert the list of messages to a formatted JSON string for logging
        agent_messages = json.dumps(
            agent_messages_list,
            indent=3,
            ensure_ascii=False
        )

        logger.opt(colors=True).debug("<magenta>[AgentCaptureMiddleware] Agent "
                                      "result:\n{}</magenta>",
                                      agent_result)
        
        logger.opt(colors=True).debug("<magenta>[AgentCaptureMiddleware] Agent "
                                      "messages:\n{}</magenta>",
                                      agent_messages)
