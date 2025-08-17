

# app.py
"""
Streamlit app: Travel + Hotel multi-agent chatbot using LangGraph + "CrewAI-style" agents
Model: Groq LLaMA-3 (llama3-70b-8192 by default)

Design:
- LangGraph orchestrates, one turn per run.
- CrewAgent shim simulates CrewAI agents.
- Handoffs between agents via HANDOFF tokens.
"""

import os
import uuid
import json
from typing import Any, List, Dict

import streamlit as st
from dotenv import load_dotenv

# LangGraph
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# LangChain / Groq
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -------------------------
# ENV + UI setup
# -------------------------
load_dotenv()
st.set_page_config(page_title="Customer support AI", layout="wide")
st.title("🌍 World Hop Multi Agent Chatbot ( Make your Trip Easy ☺️)")

# API key
if "groq_api_key" in st.session_state:
    groq_key = st.session_state.groq_api_key
else:
    groq_key = os.environ.get("GROQ_API_KEY", "")

if not groq_key:
    st.warning("Enter your **GROQ_API_KEY** to continue.")
    api_input = st.text_input("GROQ API Key", type="password")
    if api_input:
        st.session_state.groq_api_key = api_input
        os.environ["GROQ_API_KEY"] = api_input
        st.rerun()
    st.stop()
else:
    st.session_state.groq_api_key = groq_key
    os.environ["GROQ_API_KEY"] = groq_key

# Model choice
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")

# -------------------------
# Tools
# -------------------------
@tool
def get_travel_recommendations(query: str) -> str:
    """Return travel destination suggestions based on keywords."""
    q = (query or "").lower()
    if "beach" in q:
        return "Beach picks: Bali, Maldives, Phuket, Goa."
    if "mountain" in q or "trek" in q:
        return "Mountain picks: Manali, Leh-Ladakh, Swiss Alps, Banff."
    if "europe" in q:
        return "Europe picks: Paris, Rome, Barcelona, Amsterdam."
    if "budget" in q or "cheap" in q:
        return "Budget picks: Vietnam, Bali (off-season), Eastern Europe."
    return "General suggestions: beaches, mountains, cultural cities, parks."

@tool(return_direct=True)
def handoff_to_hotel_advisor() -> str:
    """Signal to switch to the hotel advisor agent."""
    return "HANDOFF:hotel_advisor"

@tool(return_direct=True)
def handoff_to_travel_advisor() -> str:
    """Signal to switch to the travel advisor agent."""
    return "HANDOFF:travel_advisor"

@tool
def get_hotel_recommendations(location: str, stars: int = 3) -> str:
    """Return hotel suggestions for a given city/area."""
    loc = (location or "").lower()
    if "bali" in loc:
        return "Bali hotels: Kuta Beach Resort, Seminyak Suites, Four Seasons Bali."
    if "goa" in loc:
        return "Goa hotels: Taj Exotica, Park Hyatt Goa, budget guesthouses."
    if "paris" in loc:
        return "Paris hotels: Hotel Le Meurice, Hotel Le Six, budget hostels."
    return f"Hotels in {location}: specify city/area for better picks."

# -------------------------
# CrewAgent shim
# -------------------------
class CrewAgent:
    def __init__(self, llm: ChatGroq, system_prompt: str, name: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.name = name

    def _messages_to_prompt(self, messages: List[Any]) -> str:
        out = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "") or m.get("text", "")
            else:
                role = getattr(m, "type", None) or getattr(m, "role", "user")
                content = getattr(m, "content", None) or getattr(m, "text", "")
            out.append(f"{role.upper()}: {content}")
        return "\n".join(out)

    def run(self, messages: List[Any]) -> List[Dict[str, str]]:
        prompt_text = f"SYSTEM: {self.system_prompt}\n\nConversation:\n" + self._messages_to_prompt(messages)
        resp = self.llm.invoke([HumanMessage(content=prompt_text)])
        text = getattr(resp, "content", None)
        if text is None:
            text = str(resp)
        return [{"role": "assistant", "content": text}]

# -------------------------
# Agents
# -------------------------
llm = ChatGroq(model=GROQ_MODEL, temperature=0.0)

TRAVEL_SYSTEM = (
    "You are Travel Advisor. Understand travel preferences (dates, budget, interests). "
    "If asked about hotels, include 'HANDOFF:hotel_advisor' on a separate line."
)
HOTEL_SYSTEM = (
    "You are Hotel Advisor. Provide hotel suggestions. "
    "If asked about destinations, include 'HANDOFF:travel_advisor' on a separate line."
)

travel_agent = CrewAgent(llm, TRAVEL_SYSTEM, "travel_advisor")
hotel_agent = CrewAgent(llm, HOTEL_SYSTEM, "hotel_advisor")

# -------------------------
# LangGraph
# -------------------------
class MultiAgentState(MessagesState):
    next_agent: str

def dispatcher_node(state: MultiAgentState, config) -> Command:
    messages = state.get("messages", [])
    current = state.get("next_agent", "travel_advisor")

    if current == "hotel_advisor":
        assistant_msgs = hotel_agent.run(messages)
    else:
        assistant_msgs = travel_agent.run(messages)

    new_messages = list(messages) + assistant_msgs
    combined = "\n".join(
        m["content"] if isinstance(m, dict) else getattr(m, "content", "")
        for m in assistant_msgs
    )

    next_agent = current
    if "HANDOFF:hotel_advisor" in combined:
        next_agent = "hotel_advisor"
    elif "HANDOFF:travel_advisor" in combined:
        next_agent = "travel_advisor"

    return Command(update={"messages": new_messages, "next_agent": next_agent})

def build_graph():
    builder = StateGraph(state_schema=MultiAgentState)
    builder.add_node("dispatcher", dispatcher_node)
    builder.add_edge(START, "dispatcher")
    return builder.compile(checkpointer=MemorySaver())

graph = build_graph()

# -------------------------
# Streamlit state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_agent" not in st.session_state:
    st.session_state.next_agent = "travel_advisor"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Render history
for m in st.session_state.messages:
    role = m.get("role", "user") if isinstance(m, dict) else getattr(m, "role", "user")
    content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
    with st.chat_message(role):
        st.markdown(content)

# -------------------------
# User input
# -------------------------
user_input = st.chat_input("Ask about travel or hotels...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    initial_state = MultiAgentState(
        messages=st.session_state.messages,
        next_agent=st.session_state.next_agent
    )

    try:
        result_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )
    except Exception as e:
        st.error(f"Graph error: {e}")
        result_state = None

    if result_state:
        new_messages = (
            result_state.get("messages", st.session_state.messages)
            if isinstance(result_state, dict)
            else getattr(result_state, "messages", st.session_state.messages)
        )
        new_next_agent = (
            result_state.get("next_agent", st.session_state.next_agent)
            if isinstance(result_state, dict)
            else getattr(result_state, "next_agent", st.session_state.next_agent)
        )

        # Find assistant messages added this turn
        prev_len = len(st.session_state.messages)
        added = new_messages[prev_len:]
        assistant_text = ""
        for a in added:
            # detect assistant role
            if isinstance(a, dict):
                is_assistant = a.get("role") == "assistant"
                raw = a.get("content") if is_assistant else None
            else:
                role_attr = getattr(a, "type", None) or getattr(a, "role", None)
                is_assistant = role_attr in ("assistant", "ai")
                raw = getattr(a, "content", None) if is_assistant else None

            if not is_assistant:
                continue

            # normalize to string
            if raw is None:
                content = ""
            elif isinstance(raw, str):
                content = raw
            elif isinstance(raw, (list, dict)):
                content = json.dumps(raw, ensure_ascii=False)
            else:
                content = str(raw)

            assistant_text += content + "\n"

        assistant_text = assistant_text.strip() or "No assistant response."
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        st.session_state.messages = new_messages
        st.session_state.next_agent = new_next_agent

st.divider()
st.write(f"Next agent: **{st.session_state.next_agent}**")
