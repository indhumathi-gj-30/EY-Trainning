from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from collections import defaultdict
import time

app = FastAPI(
    title="PolicyAssist API",
    description="AI-powered Credit Policy Query Service",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://internal.policyassist.ai"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"]
)
MAX_REQUESTS = 30
TIME_WINDOW = 60
client_requests = defaultdict(list)

def validate_rate_limit(request: Request):
    ip_address = request.client.host
    current_time = time.time()

    client_requests[ip_address] = [
        timestamp
        for timestamp in client_requests[ip_address]
        if timestamp > current_time - TIME_WINDOW
    ]

    if len(client_requests[ip_address]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Request limit exceeded. Please try again later."
        )

    client_requests[ip_address].append(current_time)

class PolicyQuery(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Policy-related question"
    )
    employee_id: str
    reference_id: Optional[str] = None
    
class PolicyResponse(BaseModel):
    request_id: str
    answer: str
    status: str
    response_time_ms: float
    contains_redaction: bool
    passed_safety_check: bool

# Query endpoint
@app.post("/api/query", response_model=PolicyResponse)
async def process_query(
    payload: PolicyQuery,
    _: None = Depends(validate_rate_limit)
):
    result = pipeline.run(
        payload.question,
        user_id=payload.employee_id
    )

    metrics = result["metrics"]

    return PolicyResponse(
        request_id=result["query_id"],
        answer=result["response"],
        status=result["action"],
        response_time_ms=metrics.latency_ms,
        contains_redaction=metrics.pii_redacted,
        passed_safety_check=metrics.output_safety_safe
    )
# Health endpoint
@app.get("/health")
def service_health():
    return {
        "status": "running",
        "service": "PolicyAssist"
    }
# Metrics endpoint
@app.get("/api/metrics")
def metrics_summary():
    data = pipeline.metrics_dataframe()

    if data.empty:
        return {"total_requests": 0}

    return {
        "total_requests": len(data),
        "blocked_requests": (data["final_action"] == "block").mean(),
        "average_latency_ms": data["latency_ms"].mean(),
        "redaction_ratio": data["pii_redacted"].mean(),
        "estimated_cost_usd": data["estimated_cost_usd"].sum()
    }
