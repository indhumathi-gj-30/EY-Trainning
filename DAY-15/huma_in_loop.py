from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
class ApprovalState(TypedDict):
    application_no: str
    requested_amount: float
    applicant_details: str
    risk_index: float

    requires_manual_check: bool
    final_decision: Optional[str]
    reviewed_by: Optional[str]
    remarks: Optional[str]
MAX_AUTO_APPROVAL_AMOUNT = 1_000_000note
MAX_ALLOWED_RISK = 0.70

def assess_application(state: ApprovalState):

    amount_flag = (
        state["requested_amount"]
        >= MAX_AUTO_APPROVAL_AMOUNT
    )

    risk_flag = (
        state["risk_index"]
        >= MAX_ALLOWED_RISK
    )

    manual_check_needed = amount_flag or risk_flag

    print("\n" + "-" * 50)
    print("APPLICATION ASSESSMENT")
    print("-" * 50)

    print(f"Application : {state['application_no']}")
    print(f"Loan Amount : ₹{state['requested_amount']:,.0f}")
    print(f"Risk Index  : {state['risk_index']:.2f}")

    if manual_check_needed:
        print("Status      : Manual Review Required")
    else:
        print("Status      : Eligible For Auto Approval")

    return {
        **state,
        "requires_manual_check": manual_check_needed
    }

def auto_approval_engine(state: ApprovalState):

    print("\nSYSTEM APPROVAL ENGINE ACTIVATED")

    return {
        **state,
        "final_decision": "approved_auto",
        "reviewed_by": "SYSTEM_ENGINE",
        "remarks": "Approved automatically by workflow rules"
    }

def manual_review_node(state: ApprovalState):

    print("\nAwaiting Underwriter Review...")

    review_packet = {
        "application_no": state["application_no"],
        "amount": state["requested_amount"],
        "risk_index": state["risk_index"],
        "applicant_details": state["applicant_details"]
    }

    reviewer_response = interrupt(review_packet)

    print("\nReviewer Response Received")

    return {
        **state,
        "final_decision": reviewer_response["decision"],
        "reviewed_by": reviewer_response["reviewer"],
        "remarks": reviewer_response.get("remarks", "")
    }

def decision_router(state: ApprovalState):

    if state["requires_manual_check"]:
        return "manual_review"

    return "auto_approval"

workflow = StateGraph(ApprovalState)

workflow.add_node(
    "assessment",
    assess_application
)

workflow.add_node(
    "auto_approval",
    auto_approval_engine
)

workflow.add_node(
    "manual_review",
    manual_review_node
)

workflow.set_entry_point("assessment")

workflow.add_conditional_edges(
    "assessment",
    decision_router,
    {
        "auto_approval": "auto_approval",
        "manual_review": "manual_review"
    }
)

workflow.add_edge(
    "auto_approval",
    END
)

workflow.add_edge(
    "manual_review",
    END
)

checkpoint_store = InMemorySaver()

loan_graph = workflow.compile(
    checkpointer=checkpoint_store
)

print("Workflow Graph Ready")
loan_requests = [

    {
        "application_no": "LN-501",
        "requested_amount": 450000,
        "applicant_details":
            "Permanent employee, credit score 735",

        "risk_index": 0.22,

        "requires_manual_check": False,
        "final_decision": None,
        "reviewed_by": None,
        "remarks": None
    },

    {
        "application_no": "LN-502",
        "requested_amount": 1800000,
        "applicant_details":
            "Business owner, credit score 670",

        "risk_index": 0.55,

        "requires_manual_check": False,
        "final_decision": None,
        "reviewed_by": None,
        "remarks": None
    },

    {
        "application_no": "LN-503",
        "requested_amount": 250000,
        "applicant_details":
            "Previous repayment issues detected",

        "risk_index": 0.88,

        "requires_manual_check": False,
        "final_decision": None,
        "reviewed_by": None,
        "remarks": None
    }
]
for counter, application in enumerate(
    loan_requests,
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"PROCESSING APPLICATION #{counter}")
    print("=" * 70)

    execution_config = {
        "configurable": {
            "thread_id": f"workflow-{counter}"
        }
    }

    response = loan_graph.invoke(
        application,
        config=execution_config
    )

    if "__interrupt__" not in response:

        print("\nFINAL OUTPUT")
        print(response)

        continue

    review_payload = (
        response["__interrupt__"][0]
        .value
    )

    print("\nREVIEW PACKAGE")
    print(review_payload)

    if review_payload["risk_index"] < 0.70:

        reviewer_decision = {
            "decision": "approved_manual",
            "reviewer": "UNDERWRITER_A01",
            "remarks":
                "Financial records verified successfully"
        }

    else:

        reviewer_decision = {
            "decision": "rejected_manual",
            "reviewer": "UNDERWRITER_B09",
            "remarks":
                "Risk level exceeds approval policy"
        }

    print("\nMANUAL DECISION")
    print(reviewer_decision)

    completed_result = loan_graph.invoke(
        Command(
            resume=reviewer_decision
        ),
        config=execution_config
    )

    print("\nFINAL OUTPUT")
    print(completed_result)


try:

    from IPython.display import (
        Image,
        display
    )

    display(
        Image(
            loan_graph
            .get_graph()
            .draw_mermaid_png()
        )
    )

except Exception:
    pass