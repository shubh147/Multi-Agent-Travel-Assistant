# agents.py
"""
World-Hop Travelers: Multi-Agent Travel Assistant
Parallel fan-out across only the relevant agents, with a single concise, de-duplicated report.

Stack: Streamlit + Groq (LLaMA 3 70B) + lightweight “CrewAI-style” agents

Notes:
- No manual selector: agent detection is automatic from the user’s question.
- If the question mentions 1 agent topic → run exactly that one.
- If it mentions 2,3,4 topics → run exactly those agents in parallel.
- If it says “everything/full report/all sections” → run all agents.
- If nothing matches → fall back to a minimal overview (travel_advisor only).
- Output is cleaned to avoid repetition across sections and inside each section.
- Flights table renderer will format JSON-like results if an agent returns them.
"""

import os
import re
import json
import textwrap
from typing import Any, List, Dict, Optional, Tuple, Set
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Optional LangGraph imports kept for later expansion (not used now)
try:
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
except Exception:
    # If not installed, we ignore - LangGraph is optional
    START = MessagesState = StateGraph = MemorySaver = Command = None  # type: ignore

# LangChain / Groq (tools)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# =========================
# ENV + UI SETUP
# =========================
load_dotenv()
st.set_page_config(page_title="World-Hop Multi-Agent Assistant", layout="wide")
st.title("🌍 World-Hop Multi-Agent Travel Assistant")

# API key bootstrapping
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

# =========================
# USER PROFILE (Personalization)
# =========================
if "profile" not in st.session_state:
    st.session_state.profile = {
        "budget_usd": 1000,
        "budget_inr": 1000 * 83,   # simple fixed conversion
        "hotel_stars": 3,
        "interests": ["beach", "food"],
        "home_airport": "DEL",
        "name": "Traveler",
    }

INR_PER_USD = 83  # fixed rate (replace with live FX if you wish)

with st.sidebar:
    st.header("🧑‍💼 Your Preferences")
    st.session_state.profile["name"] = st.text_input("Name", st.session_state.profile["name"])
    usd_val = st.number_input("Max Budget (USD)", min_value=100, value=st.session_state.profile["budget_usd"], step=50)
    inr_val = usd_val * INR_PER_USD
    st.write(f"💰 Approx Budget (INR): ₹{inr_val:,.0f}")
    st.session_state.profile["budget_usd"] = usd_val
    st.session_state.profile["budget_inr"] = inr_val
    st.session_state.profile["hotel_stars"] = st.slider(
        "Preferred Hotel Stars", min_value=1, max_value=5, value=st.session_state.profile["hotel_stars"]
    )
    st.session_state.profile["home_airport"] = st.text_input("Home Airport (IATA)", st.session_state.profile["home_airport"])
    st.session_state.profile["interests"] = st.multiselect(
        "Interests",
        ["beach", "mountains", "history", "food", "nightlife", "nature", "shopping"],
        default=st.session_state.profile["interests"],
    )
    st.caption("These preferences enrich suggestions and agent prompts.")

# =========================
# MOCK DATA + MAP HELPERS (optional)
# =========================
CITY_COORDS = {
    "bali": (-8.409518, 115.188919),
    "goa": (15.2993, 74.1240),
    "paris": (48.8566, 2.3522),
    "manali": (32.2432, 77.1892),
    "rome": (41.9028, 12.4964),
    "barcelona": (41.3874, 2.1686),
    "amsterdam": (52.3676, 4.9041),
    "phuket": (7.8804, 98.3923),
    "maldives": (3.2028, 73.2207),
    "leh": (34.1526, 77.5770)
}

def map_df_from_city(city: str) -> Optional[pd.DataFrame]:
    c = (city or "").strip().lower()
    if c in CITY_COORDS:
        lat, lon = CITY_COORDS[c]
        return pd.DataFrame([{"lat": lat, "lon": lon, "city": city.title()}])
    return None

# =========================
# TOOLS (Mock, swap with real APIs later)
# All @tool functions must have docstrings so LangChain's @tool factory won't raise errors.
# =========================

@tool
def get_travel_recommendations(query: str) -> str:
    """
    Personalized destination ideas based on user query and session profile.
    Returns a short list of recommended destinations (comma-separated).
    """
    q = (query or "").lower()
    prof = st.session_state.profile
    budget = prof.get("budget_usd", 1000)
    interests = prof.get("interests", [])

    picks = []
    if "beach" in q or "beach" in interests:
        picks += ["Bali", "Maldives", "Phuket", "Goa"]
    if "mountain" in q or "trek" in q or "mountains" in interests:
        picks += ["Manali", "Leh-Ladakh", "Swiss Alps", "Banff"]
    if "europe" in q:
        picks += ["Paris", "Rome", "Barcelona", "Amsterdam"]
    if "budget" in q or budget < 800:
        picks += ["Vietnam", "Bali (off-season)", "Budapest", "Prague"]

    if not picks:
        picks = ["Beaches, mountains, cultural cities, national parks"]
    unique_picks = ", ".join(dict.fromkeys(picks))
    return f"Suggested destinations (for ~${budget}, ~₹{budget*INR_PER_USD:,.0f}): {unique_picks}."

@tool
def get_hotel_recommendations(location: str, stars: int = 3) -> str:
    """
    Provide 2-5 hotel suggestions for a city and star rating (mock prices).
    Args:
        location: city name
        stars: desired stars
    Returns:
        A short multi-option string with price bands in USD and INR.
    """
    loc = (location or "").lower().strip()
    s = stars or st.session_state.profile.get("hotel_stars", 3)
    sample = {
        "bali": ["Kuta Beach Resort", "Seminyak Suites", "Four Seasons Bali"],
        "goa": ["Taj Exotica", "Park Hyatt Goa", "Budget Guesthouse"],
        "paris": ["Le Meurice", "Le Six", "St. Christopher’s Inn"],
    }
    items = sample.get(loc, [f"Hotel {loc.title()} Central", f"{s}-star Boutique {loc.title()}"])
    priced_list = []
    for h in items:
        usd = random.randint(50, 400)
        priced_list.append(f"{h} — {usd} USD/night (~₹{usd*INR_PER_USD:,.0f}), ~{s}★")
    return f"Hotels in {location.title()}: " + "; ".join(priced_list)

@tool
def search_flights(origin: str, destination: str, depart_date: str = "") -> str:
    """
    Return mock flight search results in JSON string form.
    Args:
        origin: origin IATA (defaults to profile home airport)
        destination: destination IATA
        depart_date: ISO date (YYYY-MM-DD) or blank -> today
    Returns:
        JSON string: { origin, destination, date, results: [ {airline, price, stops, duration}, ... ] }
    """
    origin = (origin or st.session_state.profile.get("home_airport", "DEL")).upper()
    dest = (destination or "BOM").upper()
    depart = depart_date or datetime.now().date().isoformat()
    options = [
        {"airline": "WorldHop Air", "price": random.randint(150, 800), "stops": 0, "duration": "5h 40m"},
        {"airline": "SkyLines", "price": random.randint(120, 600), "stops": 1, "duration": "7h 10m"},
        {"airline": "BudgetFly", "price": random.randint(90, 450), "stops": 2, "duration": "11h 35m"},
    ]
    return json.dumps({"origin": origin, "destination": dest, "date": depart, "results": options}, ensure_ascii=False)

@tool
def get_weather(city: str) -> str:
    """
    Return mock short weather line for a city.
    Args:
        city: city name
    Returns:
        Single-line weather summary (mock).
    """
    c = (city or "").title()
    temp = random.randint(18, 34)
    cond = random.choice(["Sunny", "Cloudy", "Light Rain", "Clear"])
    return f"Weather in {c}: {cond}, {temp}°C (mock)."

@tool
def get_events(city: str) -> str:
    """
    Return 2-3 mock upcoming events for a city.
    Args:
        city: city name
    Returns:
        Single-line events list.
    """
    c = (city or "").title()
    events = [f"{c} Food Festival", f"{c} Summer Music Night", f"{c} Cultural Heritage Walk"]
    return "Upcoming events: " + ", ".join(events)

@tool
def budget_estimator(destination: str, nights: int = 5, star: int = 3) -> str:
    """
    Provide a mock budget estimate for a destination.
    Args:
        destination: city or country name
        nights: # nights
        star: hotel star rating
    Returns:
        A concise USD + INR estimate line (mock).
    """
    base_flight = random.randint(150, 700)
    hotel_per_night = 40 + star * 30
    activities = 25 * nights
    total_usd = base_flight + hotel_per_night * nights + activities
    total_inr = total_usd * INR_PER_USD
    return f"Estimated total for {destination} ({nights} nights, ~{star}★): ~${total_usd} USD (~₹{total_inr:,.0f}) (mock)."

# Handoff tools retained for compatibility (each has a docstring)
@tool(return_direct=True, description="Handoff to hotel advisor agent.")
def handoff_to_hotel_advisor() -> str:
    """Return a standardized handoff token to hotel advisor."""
    return "HANDOFF:hotel_advisor"

@tool(return_direct=True, description="Handoff to travel advisor agent.")
def handoff_to_travel_advisor() -> str:
    """Return a standardized handoff token to travel advisor."""
    return "HANDOFF:travel_advisor"

@tool(return_direct=True, description="Handoff to flight advisor agent.")
def handoff_to_flight_advisor() -> str:
    """Return a standardized handoff token to flight advisor."""
    return "HANDOFF:flight_advisor"

@tool(return_direct=True, description="Handoff to visa advisor agent.")
def handoff_to_visa_advisor() -> str:
    """Return a standardized handoff token to visa advisor."""
    return "HANDOFF:visa_advisor"

@tool(return_direct=True, description="Handoff to local guide agent.")
def handoff_to_local_guide() -> str:
    """Return a standardized handoff token to local guide."""
    return "HANDOFF:local_guide"

@tool(return_direct=True, description="Handoff to itinerary planner agent.")
def handoff_to_itinerary_planner() -> str:
    """Return a standardized handoff token to itinerary planner."""
    return "HANDOFF:itinerary_planner"

@tool(return_direct=True, description="Handoff to budget advisor agent.")
def handoff_to_budget_advisor() -> str:
    """Return a standardized handoff token to budget advisor."""
    return "HANDOFF:budget_advisor"

# =========================
# SENTIMENT (for tone)
# =========================
def simple_sentiment(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["angry", "frustrated", "bad", "upset", "terrible"]):
        return "empathetic"
    if any(w in t for w in ["please", "thanks", "thank you", "great", "love"]):
        return "cheerful"
    return "neutral"

# =========================
# CrewAgent (thin wrapper over LLM invocation)
# =========================
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
                content = str(m.get("content", "") or m.get("text", "")) or ""
            else:
                role = getattr(m, "type", None) or getattr(m, "role", "user")
                content = str(getattr(m, "content", None) or getattr(m, "text", "")) or ""
            out.append(f"{role.upper()}: {content}")
        return "\n".join(out)

    def run(self, messages: List[Any]) -> str:
        """
        Build the prompt using system prompt + conversation and call the LLM.
        Returns the assistant text (string).
        """
        prompt_text = f"SYSTEM: {self.system_prompt}\n\nConversation:\n" + self._messages_to_prompt(messages)
        resp = self.llm.invoke([HumanMessage(content=prompt_text)])
        text = getattr(resp, "content", None)
        return text if text is not None else str(resp)

# =========================
# Agent Prompts
# =========================
def profile_context() -> str:
    p = st.session_state.profile
    return (
        f"User name: {p.get('name')}\n"
        f"Budget: ${p.get('budget_usd')} (~₹{p.get('budget_inr'):,.0f})\n"
        f"Preferred hotel stars: {p.get('hotel_stars')}\n"
        f"Interests: {', '.join(p.get('interests', []))}\n"
        f"Home airport: {p.get('home_airport')}\n"
    )

def tone_instruction(user_text: str) -> str:
    s = simple_sentiment(user_text)
    if s == "empathetic":
        return "Use a calm, empathetic tone in English."
    if s == "cheerful":
        return "Use a warm, positive tone in English."
    return "Use a professional, friendly tone in English."

def base_system(role_desc: str, user_text: str) -> str:
    return (
        f"You are {role_desc}.\n"
        f"{profile_context()}\n"
        f"{tone_instruction(user_text)}\n"
        "Output must be concise and well-structured Markdown with clear headings and bullet points.\n"
        "Do not repeat generic tips or restate the user question. Avoid duplicate bullets.\n"
        "Prefer mini-tables where helpful. Keep numbers coherent (mock data ok).\n"
    )

# Initialize LLM
llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)

# System prompt builders (lambdas that receive last user text)
TRAVEL_SYSTEM = lambda last: base_system(
    "Travel Advisor. Understand preferences and propose destinations (macro overview).", last
)
HOTEL_SYSTEM = lambda last: base_system(
    "Hotel Advisor. Suggest 2–5 hotels per city with price band and why they fit.", last
)
FLIGHT_SYSTEM = lambda last: base_system(
    "Flight Advisor. Compare 2–4 flight options with stops and durations. If returning JSON, keep keys minimal.", last
)
VISA_SYSTEM = lambda last: base_system(
    "Visa & Documents Advisor. Provide high-level guidance and succinct checklists (mock).", last
)
LOCAL_GUIDE_SYSTEM = lambda last: base_system(
    "Local Guide. Curate must-see spots, food streets, and neighborhoods.", last
)
ITINERARY_SYSTEM = lambda last: base_system(
    "Itinerary Planner. Create a compact day-by-day plan with short travel-time hints (mock).", last
)
BUDGET_SYSTEM = lambda last: base_system(
    "Budget Advisor. Estimate cost buckets and savings tips; keep consistent totals (mock).", last
)

# Initialize agents
travel_agent = CrewAgent(llm, TRAVEL_SYSTEM(""), "travel_advisor")
hotel_agent = CrewAgent(llm, HOTEL_SYSTEM(""), "hotel_advisor")
flight_agent = CrewAgent(llm, FLIGHT_SYSTEM(""), "flight_advisor")
visa_agent = CrewAgent(llm, VISA_SYSTEM(""), "visa_advisor")
local_guide_agent = CrewAgent(llm, LOCAL_GUIDE_SYSTEM(""), "local_guide")
itinerary_agent = CrewAgent(llm, ITINERARY_SYSTEM(""), "itinerary_planner")
budget_agent = CrewAgent(llm, BUDGET_SYSTEM(""), "budget_advisor")

AGENTS: Dict[str, CrewAgent] = {
    "travel_advisor": travel_agent,
    "hotel_advisor": hotel_agent,
    "flight_advisor": flight_agent,
    "visa_advisor": visa_agent,
    "local_guide": local_guide_agent,
    "itinerary_planner": itinerary_agent,
    "budget_advisor": budget_agent,
}

# =========================
# Intent → relevant agent set (strict)
# =========================
INTENT_KEYWORDS = [
    (["hotel", "stay", "resort", "hostel", "accommodation", "lodging"], "hotel_advisor"),
    (["flight", "fly", "airline", "fare", "plane", "ticket"], "flight_advisor"),
    (["visa", "passport", "documentation", "immigration"], "visa_advisor"),
    (["itinerary", "plan", "schedule", "day-by-day", "days"], "itinerary_planner"),
    (["attraction", "activity", "restaurant", "guide", "things to do", "places to visit", "local"], "local_guide"),
    (["budget", "cost", "price", "spend", "cheap", "expensive"], "budget_advisor"),
    (["destination", "travel", "trip", "where should i go", "suggest places"], "travel_advisor"),
]

ALL_AGENT_NAMES: List[str] = list(AGENTS.keys())

BROAD_TRIGGERS = ["everything", "full report", "complete report", "cover all", "all details", "all sections", "entire plan", "all agents"]

def detect_intents(user_text: str) -> Set[str]:
    """
    Return the set of relevant agent names based on the query.
    - If 'everything' (or similar) appears -> all agents.
    - Else return exactly the matched agents (1..N).
    - If none matched -> fallback to {'travel_advisor'}.
    """
    t = (user_text or "").lower()
    # All agents if explicitly asked
    if any(bt in t for bt in BROAD_TRIGGERS):
        return set(ALL_AGENT_NAMES)

    hits: Set[str] = set()
    for keys, agent in INTENT_KEYWORDS:
        if any(k in t for k in keys):
            hits.add(agent)

    if hits:
        return hits

    # Gentle fallback: minimal overview if nothing matches
    return {"travel_advisor"}

# =========================
# Parallel fan-out executor
# =========================
def run_agents_in_parallel(messages: List[Any], agent_names: Set[str]) -> Dict[str, str]:
    """
    Run the given agents in parallel and return a dict mapping agent_name -> output text.
    """
    results: Dict[str, str] = {}

    def _call(agent_key: str) -> Tuple[str, str]:
        try:
            text = AGENTS[agent_key].run(messages)
            return agent_key, text
        except Exception as e:
            return agent_key, f"(Error running {agent_key}: {e})"

    with ThreadPoolExecutor(max_workers=min(len(agent_names), 8)) as ex:
        futures = [ex.submit(_call, name) for name in agent_names]
        for f in as_completed(futures):
            k, v = f.result()
            results[k] = v
    return results

# =========================
# Output Cleaning & De-duplication
# =========================
SECTION_LABELS = {
    "travel_advisor": "Destinations & Overview",
    "hotel_advisor": "Hotels",
    "flight_advisor": "Flights",
    "visa_advisor": "Visa & Documents",
    "local_guide": "Local Guide (Attractions & Food)",
    "itinerary_planner": "Itinerary (Day-by-Day)",
    "budget_advisor": "Budget & Cost Breakdown",
}

SECTION_ORDER = [
    "travel_advisor",
    "flight_advisor",
    "hotel_advisor",
    "local_guide",
    "itinerary_planner",
    "budget_advisor",
    "visa_advisor",
]

def _sanitize_block(md: str) -> str:
    """
    Remove headings and boilerplate, deduplicate identical lines and long generic tips.
    Keeps relative structure and bullets where present.
    """
    if not md:
        return ""
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]

    # Drop patterns (headings / TL;DR)
    drop_patterns = [
        r"^#.+", r"^##.+", r"^###.+",
        r"^tl;dr", r"^tldr", r"^summary:?", r"this is an auto-compiled report"
    ]
    cleaned = []
    for ln in lines:
        if any(re.match(pat, ln, flags=re.I) for pat in drop_patterns):
            continue
        cleaned.append(ln)

    # De-duplicate identical lines (case-insensitive)
    seen = set()
    deduped = []
    for ln in cleaned:
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ln)

    # Trim very long trailing generic tips heuristically
    trimmed = []
    for ln in deduped:
        if len(ln) > 700 and ("tips" in ln.lower() or "advice" in ln.lower()):
            continue
        trimmed.append(ln)

    return "\n".join(trimmed).strip()

def _md_table_from_flights(json_text: str) -> Optional[str]:
    """
    Attempt to parse a JSON flight result (from search_flights) and render a markdown table.
    Returns None if parsing fails.
    """
    try:
        data = json.loads(json_text)
        rows = data.get("results", [])
        if not isinstance(rows, list) or not rows:
            return None
        lines = ["| Airline | Price (USD) | Stops | Duration |", "|---|---:|---:|---|"]
        for r in rows:
            lines.append(
                f"| {r.get('airline','')} | {r.get('price','')} | {r.get('stops','')} | {r.get('duration','')} |"
            )
        meta = f"**Route**: {data.get('origin','?')} → {data.get('destination','?')}  &nbsp;&nbsp; **Date**: {data.get('date','?')}"
        return meta + "\n\n" + "\n".join(lines)
    except Exception:
        return None

def _build_tldr(selected: Set[str]) -> str:
    """
    Build a concise TL;DR bullet list depending on which agents are selected.
    """
    order = [k for k in SECTION_ORDER if k in selected]
    bits = []
    if "travel_advisor" in order:
        bits.append("- **Where**: Focused destination shortlist matched to your prefs.")
    if "flight_advisor" in order:
        bits.append("- **Flights**: Compare non-stop vs 1-stop with durations.")
    if "hotel_advisor" in order:
        bits.append(f"- **Stay**: Target ~{st.session_state.profile.get('hotel_stars',3)}★ options by area & value.")
    if "local_guide" in order:
        bits.append("- **Do**: Signature sights + a few local, low-crowd picks.")
    if "itinerary_planner" in order:
        bits.append("- **Plan**: Day-by-day with travel-time hints.")
    if "budget_advisor" in order:
        bits.append("- **Budget**: Clear buckets + 10–15% buffer.")
    if "visa_advisor" in order:
        bits.append("- **Docs**: High-level checklist & pointers.")
    return "\n".join(bits)

def assemble_markdown_report(
    user_query: str,
    parallel_outputs: Dict[str, str],
    selected_agents: Set[str],
) -> str:
    """
    Build a single markdown report from agent outputs, cleaned and ordered.
    Only sections for selected_agents are included.
    """
    parts: List[str] = []
    header = f"{user_query}\n\n"
    parts.append(header)

    # TL;DR only if more than one section to avoid redundancy
    if len(selected_agents) > 1:
        tldr = _build_tldr(selected_agents)
        if tldr:
            parts.append("\n---\n### TL;DR\n" + tldr + "\n")

    # Render sections in fixed order but only for selected agents
    for agent_key in [k for k in SECTION_ORDER if k in selected_agents]:
        label = SECTION_LABELS.get(agent_key, agent_key)
        raw = parallel_outputs.get(agent_key, "")
        if not raw:
            continue

        parts.append(f"\n## {label}\n")

        # Special render for flights if content looks like JSON
        flights_md = None
        if agent_key == "flight_advisor":
            flights_md = _md_table_from_flights(raw)
        if flights_md:
            parts.append(flights_md)
        else:
            parts.append(_sanitize_block(raw))

    # Final compact footer
    parts.append("\n> Note: Data is illustrative/mocked for demo. Replace tools with live APIs for real prices & availability.")
    report = "\n".join(parts).strip()

    # Remove accidental duplicate blank lines
    report = re.sub(r"\n{3,}", "\n\n", report)
    return report

# =========================
# Orchestrator (single entry)
# =========================
def multi_agent_report(messages: List[Any]) -> str:
    """
    Main orchestrator: extract last user text, detect agents, run them in parallel,
    assemble and return a cleaned markdown report.
    """
    # Extract last user message
    last_user: str = ""
    for m in reversed(messages):
        if (isinstance(m, dict) and m.get("role") == "user") or getattr(m, "role", None) == "user":
            last_user = str(m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
            break

    # Update agent system prompts using latest tone/context
    travel_agent.system_prompt = TRAVEL_SYSTEM(last_user)
    hotel_agent.system_prompt = HOTEL_SYSTEM(last_user)
    flight_agent.system_prompt = FLIGHT_SYSTEM(last_user)
    visa_agent.system_prompt = VISA_SYSTEM(last_user)
    local_guide_agent.system_prompt = LOCAL_GUIDE_SYSTEM(last_user)
    itinerary_agent.system_prompt = ITINERARY_SYSTEM(last_user)
    budget_agent.system_prompt = BUDGET_SYSTEM(last_user)

    # Detect exactly which agents to run
    agent_set = detect_intents(last_user)

    # Run only the selected agents in parallel
    outputs = run_agents_in_parallel(messages, agent_set)

    # Assemble a concise, de-duplicated report
    report = assemble_markdown_report(last_user, outputs, agent_set)
    return report

# =========================
# CHAT UI (Input at the bottom)
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_prompt = st.chat_input(
    "Ask your question. Example: 'Flights and hotels for a 7-day Bali trip in Nov from DEL' "
    "or 'Full report for Japan in Oct (flights, hotels, itinerary, budget, visa)'."
)

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    report_markdown = multi_agent_report(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": report_markdown})
    st.rerun()
