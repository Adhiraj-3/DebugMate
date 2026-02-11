
"""
Main SRE Agent module
Creates and configures the agent with all tools and memory
"""

import uuid
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from core.ai_gateway_llm import AIGatewayLLM
from core.config import AI_GATEWAY_CONFIG
from core.prompt import build_system_prompt
from core.utils import sanitize_pii
from tools import (
    search_logstore,
    search_codebase,
    search_admin_apis,
    search_aws_cli,
    search_panic_logs,
    retrieve_stored_logs,
    set_session_context,
)

# ================== STATE DEFINITION ==================

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[list, add_messages]


# ================== CREATE AGENT ==================

# Configure LLM using AI Gateway
llm = AIGatewayLLM(**AI_GATEWAY_CONFIG)

# Build dynamic system prompt with service configs
SYSTEM_PROMPT = build_system_prompt()

# Create prompt template with system message
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

memory = MemorySaver()

agent = create_react_agent(
    llm,
    tools=[search_logstore, search_codebase, search_admin_apis, search_aws_cli, search_panic_logs, retrieve_stored_logs],
    checkpointer=memory,
    prompt=prompt
)


# ================== BUILD WORKFLOW ==================

# Build the simple graph - just agent node
workflow = StateGraph(AgentState)

# Add agent node
workflow.add_node("agent", agent)

# Simple flow: START → agent → END
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

print("✅ Agent graph created with routing logic!")

def run_agent_with_session(user_query: str, session_id: str = None) -> tuple:
    """
    Execute the agent with session-based memory.

    Args:
        user_query: The user's question or request
        session_id: Optional session ID for conversation continuity (auto-generated if not provided)

    Returns:
        tuple: (final_response, session_id)
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Configuration for the agent
    # thread_id is used by MemorySaver to track conversation history
    config = {
        "configurable": {
            "thread_id": session_id
        },
        # LLM will decide recursion depth naturally - no hard limit imposed
        "recursion_limit": 100  # Safety limit, but LLM decides when to stop
    }

    # Set logstore session context so log embeddings persist within this thread
    set_session_context(session_id)

    print(f"🔄 Session ID: {session_id}")

    # Initialize response variable
    last_ai_message = None

    # Invoke the agent - it will automatically:
    # 1. Load previous conversation from memory (if session exists)
    # 2. Process user query
    # 3. Call tools as needed (LLM decides which and when)
    # 4. Continue until LLM decides it has enough information
    # 5. Save conversation to memory
    for step in agent.stream(
        {"messages": [HumanMessage(content=user_query)]},
        config,
        stream_mode="values",
    ):
        message = step["messages"][-1]
        
        # Stream logging (optional)
        message.pretty_print()

        # Capture AI messages
        if isinstance(message, AIMessage):
            # Skip tool-calling AI messages
            if message.tool_calls:
                continue
            # Candidate final answer
            last_ai_message = message

    # After streaming completes
    if last_ai_message and last_ai_message.content:
        final_response = last_ai_message.content
        print("\n✅ FINAL RESPONSE SELECTED")
        print(f"✅ Length: {len(final_response)} chars")
    else:
        print("⚠️ WARNING: No suitable final AI response found!")
        final_response = "I apologize, but I encountered an issue processing your request. Please try again."

    # ===== PII SANITIZATION =====
    # Make a final LLM call (via AI Gateway) to strip personal information
    print("🔒 Running PII sanitization on agent response (via AI Gateway)...")
    final_response = sanitize_pii(final_response)
    print(f"✅ Post-sanitization length: {len(final_response)} chars")

    return final_response, session_id
