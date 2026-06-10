from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.purchases import router as purchases_router

app = FastAPI(
    title="Purchase Management API",
    description=(
        "End-to-end FastAPI demo with nested Pydantic models and the "
        "extension response pattern."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(purchases_router)


@app.get("/", tags=["Health"])
def check_status() -> dict:
    return {"status": "success", "message": "Purchase Management API is running "}