import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

def initialize_config_store():
    """Appends regional keys to the local storage environment securely."""
    target_store = ".env"
    auth_credential = "GORQ KEY"
    with open(target_store, "a") as store_writer:
     store_writer.write(f'GROQ_API_KEY="{auth_credential}"\n')
    load_dotenv(override=True)
    print("Local setup successful. Environment states synchronized.")
initialize_config_store()
class ProcessWorkflowState(TypedDict):
    primary_intent: str
    allocated_milestones: List[str]
    captured_logs: List[str]
    rejection_summary: str
    validation_status: bool
    processing_loops: int
def strategy_decomposition_node(state: ProcessWorkflowState) -> ProcessWorkflowState:
    """Deconstructs the global intent into isolated, granular execution milestones."""
    
    inference_engine = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    core_prompt = (
        "You are a planning agent. Break the user's goal into "
        "at most 5 concrete, actionable tasks. Respond ONLY with a "
        "valid JSON array of strings. No preamble, no markdown."
    )

    conversation_history = [
        SystemMessage(content=core_prompt),
        HumanMessage(content=f"Goal: {state['primary_intent']}")
    ]
    
    model_response = inference_engine.invoke(conversation_history).content.strip()

    try:
        clean_json_string = model_response.replace("```json", "").replace("```", "").strip()
        extracted_steps = json.loads(clean_json_string)
    except json.JSONDecodeError:
        extracted_steps = [model_response]

    print(f"\n[Decomposition Engine] Identified {len(extracted_steps)} sequence points:")
    for position, step in enumerate(extracted_steps):
        print(f"  [{position + 1}] -> {step}")

    return {**state, "allocated_milestones": extracted_steps}

if __name__ == "__main__":
    Active_Session_State: ProcessWorkflowState = {
        "primary_intent": "Research and summarise the top 3 trends in agriculture for 2025",
        "allocated_milestones": [],
        "captured_logs": [],
        "rejection_summary": "",
        "validation_status": False,
        "processing_loops": 0
    }
    
    strategy_decomposition_node(Active_Session_State)