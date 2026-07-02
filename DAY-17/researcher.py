import operator
import os
from typing import Annotated, List, TypedDict, Literal, Dict, Any
from pydantic import BaseModel, Field


class AnalystPipelineState(TypedDict):
    """STM Context Schema: Data tracking payload inside active state machine graph."""
    objective_target: str
    curated_intelligence: Annotated[List[str], operator.add]
    synthesized_brief: str
    allocated_route: str
    exception_tally: int
    compliance_directives: str

class OrchestrationRouter(BaseModel):
    """Validation schema defining dynamic edge conditions for supervisor control."""
    target_specialist: Literal["market_scanner", "content_compositor", "TERMINATION_GATE"] = Field(
        description="The next execution node designated to parse telemetry state."
    )
    operational_orders: str = Field(description="Strict inline feedback or parameters for the worker agent.")
    requires_human_signoff: bool = Field(description="True drops system state to breakpoint pause for review.")


# ─── 2. PLATFORM CONFIGURATION AND AGENT INITIALIZATION ─────────────────

# Resolving local package installations via notebook environment
# !pip install langchain_groq langchain_community

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# Pulling environment API secrets cleanly
CREDENTIAL_GROQ = os.getenv('GROQ_API_KEY', "YOUR_SECURE_GROQ_KEY")
CREDENTIAL_TAVILY = os.getenv('TAVILY_API_KEY', "YOUR_SECURE_TAVILY_KEY")

# Initializing language brains and dynamic web access matrices
agentic_brain = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=CREDENTIAL_GROQ)
search_intelligence_tool = TavilySearchResults(k=2, tavily_api_key=CREDENTIAL_TAVILY)


# ─── 3. DECENTRALIZED MULTI-AGENT STATE WORKERS ─────────────────────────

def market_scanner(state: AnalystPipelineState):
    """Researcher Node: Queries modern web search data indices into active short term state."""
    print("🚀 Market Scanner Agent: Querying deep financial database frameworks...")
    search_keyword = state['objective_target']
    raw_telemetry = search_intelligence_tool.invoke(search_keyword)
    print(f"Extraction Completed: {str(raw_telemetry)[:110]}...")
    return {"curated_intelligence": [str(raw_telemetry)], "exception_tally": 0}

def content_compositor(state: AnalystPipelineState):
    """Writer Node: Transforms current STM data blocks into refined executive briefs."""
    print("✍️ Content Compositor Agent: Generating formal intelligence briefing...")
    grounded_context = "\n".join(state['curated_intelligence'])
    
    prompt = f"""
    You are the Senior WealthIntel Content Compositor. 
    Review the background short-term data metrics collected below:
    {grounded_context}
    
    Task Objective: Draft a concise business report regarding: '{state['objective_target']}'
    Include structural formatting, core metrics, and metadata citations at the footer.
    """
    completion = agentic_brain.invoke(prompt)
    return {"synthesized_brief": completion.content}

def governance_supervisor(state: AnalystPipelineState):
    """
    Supervisor Node: Coordinates transactional pathways and asserts conditional edges.
    Evaluates current progress to route next functional machine state.
    """
    print("🧠 Governance Supervisor Node: Auditing runtime transactional state...")
    structured_auditor = agentic_brain.with_structured_output(OrchestrationRouter)

    audit_prompt = f"""
    Review current state parameters for target: {state['objective_target']}
    Extracted Insights Cached: {len(state['curated_intelligence'])} records.
    Current Brief Snapshot: {state['synthesized_brief'][:100]}...
    
    CRITICAL COMPLIANCE DIRECTION: If 'curated_intelligence' has content records present, 
    you must output a trajectory heading towards 'content_compositor'.
    """
    print(f"Supervisor Metric: Cached Records = {len(state['curated_intelligence'])} | Current Document Length = {len(state['synthesized_brief'])}")
    compliance_verdict = structured_auditor.invoke(audit_prompt)
    
    return {
        "allocated_route": compliance_verdict.target_specialist,
        "compliance_directives": compliance_verdict.operational_orders
    }


# ─── 4. STATE GRAPH ARCHITECTURE AND STORAGE PARADIGMS ──────────────────

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Binding configuration schema to state manager graph
orchestration_graph = StateGraph(AnalystPipelineState)

# Mapping worker nodes into processing grid space
orchestration_graph.add_node("governance_supervisor", governance_supervisor)
orchestration_graph.add_node("market_scanner", market_scanner)
orchestration_graph.add_node("content_compositor", content_compositor)

# Setting system entry execution vector
orchestration_graph.set_entry_point("governance_supervisor")

# Integrating core conditional routing pathways
orchestration_graph.add_conditional_edges(
    "governance_supervisor",
    lambda dynamic_context: dynamic_context["allocated_route"],
    {
        "market_scanner": "market_scanner",
        "content_compositor": "content_compositor",
        "TERMINATION_GATE": END
    }
)

# Creating cyclic workflow loops matching structural blueprints
orchestration_graph.add_edge("market_scanner", "governance_supervisor")
orchestration_graph.add_edge("content_compositor", "governance_supervisor")

# Injecting local system memory saver for durable checkpoints & administrative pauses (LTM)[cite: 4]
runtime_checkpoint_saver = MemorySaver()
compiled_agent_framework = orchestration_graph.compile(
    checkpointer=runtime_checkpoint_saver,
    interrupt_before=["content_compositor"]  # HUMAN COMPLIANCE HOOK: Intercept pipeline state for authorization
)


# ─── 5. EXECUTION TRAJECTORY VALIDATION STREAM ─────────────────────────

execution_envelope = {"configurable": {"thread_id": "wealth_intel_run_9901"}}
initial_state_vector = {
    "objective_target": "Impact of LPU architecture on AI inference speeds", 
    "curated_intelligence": [], 
    "exception_tally": 0, 
    "synthesized_brief": ""
}

print("--- INITIATING SYSTEM STATE GRAPH WORKFLOW ---")
for response_packet in compiled_agent_framework.stream(initial_state_vector, execution_envelope, stream_mode="values"):
    if "allocated_route" in response_packet:
        print(f"State Shift Log -> Active Route Pointer: {response_packet['allocated_route']}")

# Evaluating pipeline status to see if system triggered a breakpoint loop
state_snapshot = compiled_agent_framework.get_state(execution_envelope)
if state_snapshot.next:
    print(f"\n WORKFLOW BREAKPOINT HIT (SYSTEM SUSPENDED). Target Delayed At: {state_snapshot.next}")
    print(f"Supervisor Compliance Directives: {state_snapshot.values['compliance_directives']}")

    print("\n--- RESUMING TRANSACTION ENGINE PROCESSING FROM HUMAN CHECKPOINT COMPLIANCE ---")

print("--- RESUMING GRAPH TRAJECTORY ---\n")
for response_packet in compiled_agent_framework.stream(None, execution_envelope, stream_mode="values"):
    if "allocated_route" in response_packet:
        print(f"State Shift Log -> Active Route Pointer: {response_packet['allocated_route']}")
    elif "synthesized_brief" in response_packet:
        print(f"\n--- FINAL EVALUATED BRIEFING (COMMITTED TO PERSISTENT MEMORY) ---:\n{response_packet['synthesized_brief']}")

# Rendering architectural layout representation in execution frame
from IPython.display import Image, display
try:
    display(Image(compiled_agent_framework.get_graph().draw_mermaid_png()))
except Exception:
    pass