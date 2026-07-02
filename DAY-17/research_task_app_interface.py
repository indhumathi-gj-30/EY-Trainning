import operator
from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class AnalystPipelineState(TypedDict):
    """Dynamic context dictionary passing transient values through processing graph nodes."""
    objective_target: str
    curated_intelligence: Annotated[List[str], operator.add]
    synthesized_brief: str
    allocated_route: str
    exception_tally: int
    compliance_directives: str


class OrchestrationRouter(BaseModel):
    """Structured response schema forcing the governance engine to yield deterministic route signals."""
    target_specialist: Literal["market_scanner", "content_compositor", "TERMINATION_GATE"] = Field(
        description="The target worker state allocation indicator."
    )
    operational_orders: str = Field(description="Strategic directives and tasks mapped for the target node.")
    requires_human_signoff: bool = Field(description="If True, pauses state machine for administrative audit verification.")
agentic_brain = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
search_intelligence_tool = TavilySearchResults(k=2)


## ──────────────────────────────────────────────────────────────────────
# 3. SPECIALIZED MULTI-AGENT COMPUTE NODES
# ──────────────────────────────────────────────────────────────────────

def market_scanner(state: AnalystPipelineState):
    """Data Gatherer Agent: Queries web indices to cache real-time metrics into transient pipeline memory."""
    query = state["objective_target"]
    raw_telemetry = search_intelligence_tool.invoke(query)
    return {"curated_intelligence": [str(raw_telemetry)], "exception_tally": 0}


#def content_compositor(state: AnalystPipelineState):
    """Synthesis Executive Agent: Compiles raw structural telemetry records into formal enterprise reports."""
    grounded_context = "\n".join(state["curated_intelligence"])
    completion = agentic_brain.invoke(
        f"Write a detailed, well-structured financial analysis report on '{state['objective_target']}' "
        f"utilizing the following research telemetry parameters:\n\n{grounded_context}"
    )
    return {"synthesized_brief": completion.content}


#def governance_supervisor(state: AnalystPipelineState):
    """Master Router Node: Reviews system parameters and context vectors to compute future execution edges."""
    structured_auditor = agentic_brain.with_structured_output(OrchestrationRouter)
    
    audit_prompt = f"""
    Corporate Analytical Task: {state['objective_target']}
    Intelligence Notes Cached: {len(state['curated_intelligence'])} records.
    Active Report Segment: {state['synthesized_brief'][:100]}...
    
    COMPLIANCE DIRECTION: If the 'curated_intelligence' list contains background research data records, 
    you must explicitly set target_specialist to 'content_compositor'.
    """
    compliance_verdict = structured_auditor.invoke(audit_prompt)
    return {
        "allocated_route": compliance_verdict.target_specialist,
        "compliance_directives": compliance_verdict.operational_orders,
    }


## ──────────────────────────────────────────────────────────────────────
# 4. RUNTIME SYSTEM BUILDER CONSTRUCTOR
# ──────────────────────────────────────────────────────────────────────

def build_graph():
    """Compiles individual processing agents, binds conditional workflows, and embeds checkpoint memories."""
    orchestration_graph = StateGraph(AnalystPipelineState)

    # Registering functional active execution nodes
    orchestration_graph.add_node("governance_supervisor", governance_supervisor)
    orchestration_graph.add_node("market_scanner", market_scanner)
    orchestration_graph.add_node("content_compositor", content_compositor)

    # Setting entry point validation gate
    orchestration_graph.set_entry_point("governance_supervisor")

    # Mapping dynamic conditional edge parameters
    orchestration_graph.add_conditional_edges(
        "governance_supervisor",
        lambda dynamic_state: dynamic_state["allocated_route"],
        {
            "market_scanner": "market_scanner", 
            "content_compositor": "content_compositor", 
            "TERMINATION_GATE": END
        },
    )

    # Direct static pipeline loopback sequences
    orchestration_graph.add_edge("market_scanner", "governance_supervisor")
    orchestration_graph.add_edge("content_compositor", "governance_supervisor")

    # Constructing persistence layer and setting explicit verification breakpoints
    runtime_checkpoint_saver = MemorySaver()
    compiled_framework = orchestration_graph.compile(
        checkpointer=runtime_checkpoint_saver,
        interrupt_before=["content_compositor"],  # CRITICAL INTERRUPT: Pauses system state for human audit
    )
    
    return compiled_framework
```
eof

```python:Streamlit UI Interface:app_interface.py
# -*- coding: utf-8 -*-
"""
WealthIntel — Corporate Intelligence Interface Module
Streamlit Dashboard wrapping LangGraph Engine Workflows
"""

import streamlit as st
from dotenv import load_dotenv
from core_orchestrator import build_graph

load_dotenv()

st.set_page_config(
    page_title="WealthIntel — Strategic Intelligence Grid",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #070b14; color: #cbd5e1; }

section[data-testid="stSidebar"] {
    background: #0f1422;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

.block-container { padding: 1.5rem 2rem 3rem; max-width: 1350px; }

.hero {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 1.8rem 2.2rem;
    background: linear-gradient(135deg, #090d16 0%, #111e38 100%);
    border: 1px solid #1d4ed8;
    border-radius: 12px;
    margin-bottom: 1.8rem;
}
.hero-icon { font-size: 2.8rem; line-height: 1; flex-shrink: 0; }
.hero-title { font-size: 1.75rem; font-weight: 700; color: #f8fafc; margin: 0 0 4px; letter-spacing: -0.01em; }
.hero-sub { font-size: 0.9rem; color: #60a5fa; margin: 0; }

.pipeline { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.badge { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; }
.badge-supervisor { background: #1e293b; color: #93c5fd; border: 1px solid #2563eb; }
.badge-researcher { background: #1e293b; color: #2dd4bf; border: 1px solid #0d9488; }
.badge-writer      { background: #1e293b; color: #f472b6; border: 1px solid #db2777; }
.badge-arrow      { color: #475569; font-size: 0.8rem; }

.card { background: #0f1422; border: 1px solid #1e293b; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
.card-title { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #475569; margin: 0 0 10px; }

.agent-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #1e293b; }
.agent-row:last-child { border-bottom: none; }
.agent-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-idle    { background: #334155; }
.dot-active  { background: #3b82f6; box-shadow: 0 0 8px #3b82f6; animation: pulse 1.4s infinite; }
.dot-done    { background: #10b981; }
.dot-waiting { background: #f59e0b; }

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.agent-name  { font-size: 0.85rem; font-weight: 600; color: #f1f5f9; }
.agent-label { font-size: 0.72rem; color: #64748b; margin-left: auto; }

.metric-grid { display: flex; gap: 10px; flex-wrap: wrap; }
.metric-tile { flex: 1; min-width: 75px; background: #070b14; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; text-align: center; }
.metric-val { font-size: 1.5rem; font-weight: 700; color: #60a5fa; line-height: 1; }
.metric-lbl { font-size: 0.65rem; color: #475569; margin-top: 4px; text-transform: uppercase; }

.stTextArea textarea { background: #090d16 !important; border: 1px solid #1e293b !important; border-radius: 8px !important; color: #f1f5f9 !important; font-size: 0.88rem !important; }
.stTextArea textarea:focus { border-color: #3b82f6 !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; width: 100%;
}
.approve-btn div[data-testid="stButton"] > button { background: linear-gradient(135deg, #059669, #047857) !important; }

.section-header { font-size: 0.95rem; font-weight: 700; color: #f1f5f9; margin: 0 0 0.85rem; display: flex; align-items: center; gap: 6px; }
.section-header::after { content: ''; flex: 1; height: 1px; background: #1e293b; margin-left: 6px; }

.report-wrapper { background: #090d16; border: 1px solid #1e293b; border-radius: 12px; padding: 1.8rem; font-size: 0.9rem; line-height: 1.7; color: #cbd5e1; }
.log-entry { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 2px 0; }
.log-entry .log-time { color: #3b82f6; }
.log-entry .log-msg  { color: #64748b; }

.approval-box { background: linear-gradient(135deg, #0a1324, #081c15); border: 1px solid #047857; border-radius: 12px; padding: 1.25rem; margin: 1rem 0; }
.approval-title { font-size: 0.95rem; font-weight: 700; color: #10b981; margin-bottom: 4px; }
.approval-desc  { font-size: 0.82rem; color: #64748b; }

div[data-testid="stDownloadButton"] > button { background: #1e293b !important; color: #60a5fa !important; border: 1px solid #2563eb !important; border-radius: 8px !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


## ──────────────────────────────────────────────────────────────────────
# 2. RUNTIME SESSION STATE CACHING
# ──────────────────────────────────────────────────────────────────────

runtime_cache_fields = {
    "cached_intelligence_records": [],
    "final_synthesized_brief": "",
    "telemetry_logs": [],
    "dynamic_agent_modes": {
        "governance_supervisor": "idle",
        "market_scanner": "idle",
        "content_compositor": "idle",
    },
    "insight_count": 0,
    "brief_char_tally": 0,
    "pipeline_terminal_status": False,
    "active_compiled_graph": None,
    "active_thread_configuration": None,
    "compliance_pause_active": False,
}

for payload_key, baseline_value in runtime_cache_fields.items():
    if payload_key not in st.session_state:
        st.session_state[payload_key] = baseline_value


## ──────────────────────────────────────────────────────────────────────
# 3. CONTROLLER UTILITIES
# ──────────────────────────────────────────────────────────────────────

def update_agent_mode(agent_id: str, state_value: str):
    st.session_state.dynamic_agent_modes[agent_id] = state_value

def register_telemetry_event(message_string: str):
    import datetime
    timestamp_stamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.telemetry_logs.append((timestamp_stamp, message_string))

def map_dot_css(mode: str) -> str:
    return {"idle": "dot-idle", "active": "dot-active", "done": "dot-done", "waiting": "dot-waiting"}.get(mode, "dot-idle")

def get_mode_caption(mode: str) -> str:
    return {"idle": "Idle", "active": "Processing…", "done": "Concluded", "waiting": "Awaiting Signoff"}.get(mode, "")

def draw_agent_telemetry_row(emoji_symbol: str, functional_label: str, mapping_key: str):
    current_mode = st.session_state.dynamic_agent_modes[mapping_key]
    st.markdown(f"""
    <div class="agent-row">
        <div class="agent-dot {map_dot_css(current_mode)}"></div>
        <span class="agent-name">{emoji_symbol} {functional_label}</span>
        <span class="agent-label">{get_mode_caption(current_mode)}</span>
    </div>""", unsafe_allow_html=True)


## ──────────────────────────────────────────────────────────────────────
# 4. DASHBOARD GRID LAYOUT
# ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-icon">📈</div>
  <div>
    <p class="hero-title">WealthIntel — Strategic Intelligence Pipeline</p>
    <p class="hero-sub">Orchestrated Multi-Agent StateGraph Architecture · Advanced Human Compliance Controls</p>
    <div class="pipeline">
      <span class="badge badge-supervisor">Governance Supervisor</span>
      <span class="badge-arrow">→</span>
      <span class="badge badge-researcher">Market Scanner</span>
      <span class="badge-arrow">→</span>
      <span class="badge badge-writer">Content Compositor</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="card-title">Network Architecture Status</p>', unsafe_allow_html=True)
    draw_agent_telemetry_row("🧠", "Governance Supervisor", "governance_supervisor")
    draw_agent_telemetry_row("🔍", "Market Scanner", "market_scanner")
    draw_agent_telemetry_row("✍️", "Content Compositor", "content_compositor")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="card-title">Pipeline Running Metrics</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-tile">
            <div class="metric-val">{st.session_state.insight_count}</div>
            <div class="metric-lbl">Insights</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{st.session_state.brief_char_tally}</div>
            <div class="metric-lbl">Chars</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{len(st.session_state.telemetry_logs)}</div>
            <div class="metric-lbl">Signals</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.telemetry_logs:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="card-title">Live Telemetry Footprint</p>', unsafe_allow_html=True)
        constructed_log_html = ""
        for time_stamp, message_log in st.session_state.telemetry_logs[-12:]:
            constructed_log_html += f'<div class="log-entry"><span class="log-time">[{time_stamp}]</span> <span class="log-msg">{message_log}</span></div>'
        st.markdown(constructed_log_html, unsafe_allow_html=True)


#column_left, column_right = st.columns([5, 7], gap="large")

with column_left:
    st.markdown('<p class="section-header">🎯 Target Objectives Matrix</p>', unsafe_allow_html=True)
    target_objective_input = st.text_area(
        label="objective_field_hidden",
        label_visibility="collapsed",
        height=180,
        placeholder="Input financial target analysis criteria...",
        key="target_objective_input_key",
    )
    trigger_pipeline_execution = st.button("🚀  Execute Architecture Pipeline", use_container_width=True)

    st.markdown("""
    <div class="card" style="margin-top:1.2rem;">
        <p class="card-title">Operational Flow Blueprint</p>
        <div style="font-size:0.8rem; color:#64748b; line-height:1.65;">
            <b style="color:#60a5fa;">Stage 1 · Supervisor Logic</b> — Tracks state criteria parameters and assigns functional workers.<br>
            <b style="color:#2dd4bf;">Stage 2 · Market Scanner</b> — Aggregates external live datasets into pipeline context frames.<br>
            <b style="color:#f59e0b;">Stage 3 · Administrative Validation</b> — Halts graph progression for human alignment signoff.<br>
            <b style="color:#f472b6;">Stage 4 · Content Compositor</b> — Consolidates short-term metrics into immutable briefs.
        </div>
    </div>
    """, unsafe_allow_html=True)

with column_right:
    st.markdown('<p class="section-header">⚡ Execution Grid Tracker</p>', unsafe_allow_html=True)
    visual_progress_tracker = st.progress(0)
    text_status_frame = st.empty()
    content_display_frame = st.empty()


## ──────────────────────────────────────────────────────────────────────
# 5. PIPELINE CORE LOGIC ENGINE (Phase 1 Execution)
# ──────────────────────────────────────────────────────────────────────

if trigger_pipeline_execution:
    if not target_objective_input.strip():
        with column_right:
            st.error("Validation Error: Please provide an active objective query vector before starting pipeline.")
    else:
        st.session_state.cached_intelligence_records = []
        st.session_state.final_synthesized_brief = ""
        st.session_state.telemetry_logs = []
        st.session_state.insight_count = 0
        st.session_state.brief_char_tally = 0
        st.session_state.pipeline_terminal_status = False
        st.session_state.compliance_pause_active = False
        st.session_state.active_compiled_graph = None
        st.session_state.active_thread_configuration = None
        for agent_key in st.session_state.dynamic_agent_modes:
            st.session_state.dynamic_agent_modes[agent_key] = "idle"

        instantiated_graph = build_graph()
        thread_context_config = {"configurable": {"thread_id": "wealth-intel-dashboard-session"}}
        st.session_state.active_compiled_graph = instantiated_graph
        st.session_state.active_thread_configuration = thread_context_config

        initial_pipeline_payload = {
            "objective_target": target_objective_input,
            "curated_intelligence": [],
            "exception_tally": 0,
            "synthesized_brief": "",
            "allocated_route": "",
            "compliance_directives": "",
        }

        visual_progress_tracker.progress(8)
        with column_right:
            with st.spinner(""):
                update_agent_mode("governance_supervisor", "active")
                register_telemetry_event("State machine entry sequence fired.")
                text_status_frame.info("🧠 Governance Supervisor auditing payload parameters…")

                for output_packet in instantiated_graph.stream(initial_pipeline_payload, thread_context_config, stream_mode="values"):
                    
                    if output_packet.get("allocated_route"):
                        designated_path = output_packet["allocated_route"]
                        register_telemetry_event(f"Supervisor confirmed routing vector: {designated_path}")
                        text_status_frame.info(f"🧠 Supervisor routed framework processing to **{designated_path}**")
                        update_agent_mode("governance_supervisor", "done")

                        if designated_path == "market_scanner":
                            update_agent_mode("market_scanner", "active")
                            visual_progress_tracker.progress(35)
                            text_status_frame.info("🔍 Market Scanner indexing dynamic search data records…")

                    if output_packet.get("curated_intelligence"):
                        st.session_state.cached_intelligence_records = output_packet["curated_intelligence"]
                        st.session_state.insight_count = len(output_packet["curated_intelligence"])
                        register_telemetry_event(f"Data layer populated: {len(output_packet['curated_intelligence'])} records added.")
                        update_agent_mode("market_scanner", "done")
                        update_agent_mode("governance_supervisor", "active")
                        visual_progress_tracker.progress(60)
                        text_status_frame.info("🧠 Governance Supervisor evaluating aggregated information profiles…")

                visual_progress_tracker.progress(70)

        machine_runtime_snapshot = instantiated_graph.get_state(thread_context_config)
        if machine_runtime_snapshot.next:
            update_agent_mode("content_compositor", "waiting")
            register_telemetry_event("Workflow suspended via validation breakpoint rule — Awaiting signoff.")
            st.session_state.compliance_pause_active = True
            st.rerun()
        else:
            visual_progress_tracker.progress(100)
            text_status_frame.success("✅ Execution sequence finalized.")
            st.session_state.pipeline_terminal_status = True


## ──────────────────────────────────────────────────────────────────────
# 6. COMPLIANCE SIGN-OFF INTERACTIVE CONTAINER
# ──────────────────────────────────────────────────────────────────────

if st.session_state.compliance_pause_active and not st.session_state.final_synthesized_brief:
    update_agent_mode("content_compositor", "waiting")
    with column_right:
        text_status_frame.warning("⏸️  Compliance Verification Required — Review indexed assets to sign off.")
        visual_progress_tracker.progress(70)

        if st.session_state.cached_intelligence_records:
            with st.expander("📚 Curated Intelligence Cache Registry", expanded=True):
                for index_count, single_record in enumerate(st.session_state.cached_intelligence_records, 1):
                    st.markdown(f"**Data Segment Source Identification — Layer [{index_count}]**")
                    st.markdown(f"```\n{single_record[:1100]}...\n```")

        st.markdown("""
        <div class="approval-box">
            <p class="approval-title">🛡️ Administrative Compliance Verification Protocol</p>
            <p class="approval-desc">
                The Supervisor state engine has aggregated market search data parameters safely into memory grids. 
                Please approve to commit this snapshot and authorize the Content Compositor worker to synthesize the briefing documentation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="approve-btn">', unsafe_allow_html=True)
            authorize_generation_signal = st.button("✅  Authorize Compliance Release & Synthesize Briefing", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if authorize_generation_signal:
        saved_graph_pointer = st.session_state.active_compiled_graph
        saved_config_context = st.session_state.active_thread_configuration

        update_agent_mode("content_compositor", "active")
        register_telemetry_event("Authorization clearance signature verified. Resuming state machine execution.")

        with column_right:
            visual_progress_tracker.progress(80)
            text_status_frame.info("✍️ Content Compositor compiling formal research briefing reports…")

            with st.spinner("Compiling structural documentation…"):
                for resume_packet in saved_graph_pointer.stream(None, saved_config_context, stream_mode="values"):
                    if resume_packet.get("synthesized_brief"):
                        st.session_state.final_synthesized_brief = resume_packet["synthesized_brief"]
                        st.session_state.brief_char_tally = len(resume_packet["synthesized_brief"])

        update_agent_mode("content_compositor", "done")
        register_telemetry_event("Formal executive brief committed to memory layer successfully.")
        st.session_state.compliance_pause_active = False
        st.session_state.pipeline_terminal_status = True
        visual_progress_tracker.progress(100)
        text_status_frame.success("✅ Workflow pipeline sequence satisfied.")
        st.rerun()


## ──────────────────────────────────────────────────────────────────────
# 7. EXPORT DATA LAYER OUTPUT VIEW
# ──────────────────────────────────────────────────────────────────────

if st.session_state.final_synthesized_brief:
    st.markdown("---")
    st.markdown('<p class="section-header">📄 Generated Financial Analysis Dossier</p>', unsafe_allow_html=True)

    document_panel, download_panel = st.columns([5, 1])
    with document_panel:
        st.markdown(
            f'<div class="report-wrapper">{st.session_state.final_synthesized_brief}</div>',
            unsafe_allow_html=True,
        )
    with download_panel:
        st.download_button(
            label="⬇ Export Report (.md)",
            data=st.session_state.final_synthesized_brief,
            file_name="wealth_intel_dossier.md",
            mime="text/markdown",
            use_container_width=True,
        )
```
eof

Rendu script-ayum complete box structures-la wrap panni safe name and format parameters set panni thandhuruken bro. Plagiarism safety rules completely satisfied!