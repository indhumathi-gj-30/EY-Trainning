import time
import uuid
import logging
import contextvars
from datetime import datetime
from collections import defaultdict
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter
import httpx

app = FastAPI()

# Standard logging configuration to handle transaction flows
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("transaction_logger")

MAX_ALLOWED_REQUESTS = 100    # Threshold request count
ROLLING_WINDOW_SECS = 60      # Monitoring window frame

# InMemory visitor registry lookup mapping
visitor_registry = defaultdict(list)

THROTTLED_EVENTS_TRACKER = Counter(
    "http_rate_limit_blocks_total",
    "Cumulative count of rejected requests exceeding threshold limits"
)


# Localized asynchronous request lifecycle storage token
trace_context_holder = contextvars.ContextVar(
    "distributed_trace_id",
    default=None
)

@app.middleware("http")
async def traffic_and_tracing_interceptor(
    incoming_request: Request,
    pipeline_next: Callable
) -> Response:

    remote_host = incoming_request.client.host
    current_timestamp = datetime.utcnow()

    # Clear stale historical request objects outside rolling window threshold
    visitor_registry[remote_host] = [
        entry_time for entry_time in visitor_registry[remote_host]
        if (current_timestamp - entry_time).total_seconds() < ROLLING_WINDOW_SECS
    ]

    # Validate client access status capability limits
    if len(visitor_registry[remote_host]) >= MAX_ALLOWED_REQUESTS:
        THROTTLED_EVENTS_TRACKER.inc()

        earliest_active_record = min(visitor_registry[remote_host])
        cooldown_duration = ROLLING_WINDOW_SECS - int(
            (current_timestamp - earliest_active_record).total_seconds()
        )

        return JSONResponse(
            status_code=429,
            content={
                "status": "Rejected",
                "reason": "Request volume exceeded operational parameters",
                "retry_after_seconds": cooldown_duration
            },
            headers={
                "Retry-After": str(cooldown_duration)
            }
        )

    visitor_registry[remote_host].append(current_timestamp)

    # ---------- Context Tracing Engine ----------
    diagnostic_token = incoming_request.headers.get(
        "X-Correlation-Id",
        str(uuid.uuid4())
    )

    trace_context_holder.set(diagnostic_token)
    execution_start_marker = time.perf_counter()

    log.info(
        f"http.transaction.initiated | Endpoint: {incoming_request.url.path} | "
        f"Verb: {incoming_request.method} | TraceID: {diagnostic_token}"
    )

    # Process cascading core lifecycle logic execution
    outgoing_response = await pipeline_next(incoming_request)

    calculated_latency_ms = round(
        (time.perf_counter() - execution_start_marker) * 1000,
        2
    )

    log.info(
        f"http.transaction.finalized | Endpoint: {incoming_request.url.path} | "
        f"Status: {outgoing_response.status_code} | Duration: {calculated_latency_ms}ms | "
        f"TraceID: {diagnostic_token}"
    )

    outgoing_response.headers["X-Correlation-Id"] = diagnostic_token

    return outgoing_response

async def forward_distributed_trace(
    outbound_payload: httpx.Request
):
    active_trace_id = trace_context_holder.get()

    if active_trace_id:
        outbound_payload.headers[
            "X-Correlation-Id"
        ] = active_trace_id


network_client_pool = httpx.AsyncClient(
    event_hooks={
        "request": [forward_distributed_trace]
    }
)


@app.get("/mock-downstream")
async def dependency_simulation_endpoint(
    received_query: Request
):
    return {
        "verified_trace_token":
            received_query.headers.get(
                "X-Correlation-Id"
            )
    }

@app.get("/test-propagation")
async def verify_pipeline_propagation():

    dependency_response = await network_client_pool.get(
        "http://localhost:8000/mock-downstream"
    )

    return {
        "status": "Trace context verified across loop boundaries",
        "nested_dependency_payload":
            dependency_response.json()
    }


print("System Extension A: Traffic Control Pipeline Ready.")
print("System Extension B: Trace Propagation Infrastructure Ready.")