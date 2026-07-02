import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class StrategyMetrics:
    user_prompt: str
    analytical_intent: str       
    target_sources: list   
    max_k: int
    block_size: int
    search_modality: str            
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ExecutionOutput:
    user_prompt: str
    strategy: StrategyMetrics
    fetched_blocks: list   
    generated_response: str
    duration_seconds: float
    computed_tokens: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AuditReport:
    user_prompt: str
    generated_response: str
    grounded_score: float
    alignment_score: float
    source_coverage_ratio: float
    passed_compliance: bool
    system_alerts: list
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class StorageEngine:
    """Enterprise secure serialization handler for session logging."""
    def __init__(self, target_directory: str = "./wealth_store"):
        self.root = Path(target_directory)
        self.root.mkdir(exist_ok=True)
        self.cache_registry = self.root / "cache_registry.pkl"
        self.vector_bin = self.root / "vector_bin.pkl"
        self.run_history = self.root / "run_history.pkl"

    def save_index(self, vector_store, data_blocks: list, metadata: dict):
        with open(self.vector_bin, "wb") as f:
            pickle.dump({"vs": vector_store, "blocks": data_blocks, "meta": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self):
        if not self.vector_bin.exists(): return None, None, None
        with open(self.vector_bin, "rb") as f:
            d = pickle.load(f)
            return d["vs"], d["blocks"], d["meta"]

    def append_history(self, record: dict):
        logs = []
        if self.run_history.exists():
            with open(self.run_history, "rb") as f: logs = pickle.load(f)
        logs.append(record)
        with open(self.run_history, "wb") as f:
            pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_history(self) -> list:
        if not self.run_history.exists(): return []
        with open(self.run_history, "rb") as f: return pickle.load(f)

class StrategyNode:
    """Classifies user payload and dynamically configures retrieval strategy."""
    RULES = {
        "comparative": ["compare", "vs", "versus", "difference"],
        "trend": ["trend", "growth", "increase", "over time", "yoy"],
        "risk": ["risk", "challenge", "threat", "uncertainty"],
        "revenue": ["revenue", "sales", "income", "profit"]
    }

    def formulate_strategy(self, user_prompt: str, runtime_config: dict) -> StrategyMetrics:
        text_lower = user_prompt.lower()
        intent = "generic_factual"
        
        for category, triggers in self.RULES.items():
            if any(word in text_lower for word in triggers):
                intent = category
                break

        k_val = 6 if intent == "comparative" else (5 if intent in ("trend", "revenue") else runtime_config.get("retrieval_k", 4))
        b_size = 256 if intent == "comparative" else 512

        sources = []
        if intent in ("revenue", "trend"): sources = ["Risk", "Products"]
        elif intent == "risk": sources = ["Risk"]
        elif intent == "comparative": sources = ["Risk", "Products", "Liquidity"]

        return StrategyMetrics(
            user_prompt=user_prompt, analytical_intent=intent,
            target_sources=sources, max_k=k_val, block_size=b_size,
            search_modality=runtime_config.get("strategy", "dense")
        )

class ExecutionCore:
    """Executes vectorized searching and compiles targeted response context."""
    PROMPT = """You are WealthIntel, an expert quantitative research system.
Answer the prompt strictly utilizing the extracted context blocks.
If data is missing, respond with "Insufficient information in the retrieved context."
Append specific document metadata at the end of the text body.

CONTEXT BASE:
{context_str}

PROMPT: {question}
RESPONSE (Include metadata):"""

    def __init__(self, vector_store, embedding_engine, language_model):
        self.vs = vector_store
        self.embed = embedding_engine
        self.llm = language_model

    def execute_pipeline(self, strategy: StrategyMetrics) -> ExecutionOutput:
        start_time = time.time()
        search_engine = self.vs.as_retriever(search_type="mmr", search_kwargs={"k": strategy.max_k, "fetch_k": strategy.max_k * 3})
        
        nodes = search_engine.invoke(strategy.user_prompt)
        blocks = [{"source": n.metadata.get("source", "unknown"), "content": n.page_content} for n in nodes]
        
        context_payload = "\n\n".join(f"[Metadata: {b['source']}]\n{b['content']}" for b in blocks)
        rendered_prompt = self.PROMPT.format(context_str=context_payload, question=strategy.user_prompt)
        
        inference = self.llm.invoke(rendered_prompt)
        text_output = inference.content if hasattr(inference, "content") else str(inference)
        
        return ExecutionOutput(
            user_prompt=strategy.user_prompt, strategy=strategy, fetched_blocks=blocks,
            generated_response=text_output, duration_seconds=round(time.time() - start_time, 3),
            computed_tokens=len(rendered_prompt.split()) + len(text_output.split())
        )

class AuditNode:
    """Performs compliance verification and statistical token overlap grading."""
    def verify_integrity(self, output: ExecutionOutput) -> AuditReport:
        alerts = []
        
        # 1. Scope Evaluation
        req = set(output.strategy.target_sources)
        ret = {b["source"].split("_")[2] if "_" in b["source"] else b["source"] for b in output.fetched_blocks}
        cov = sum(1 for s in req if any(s in r for r in ret)) / len(req) if req else 1.0
        if cov < 0.5: alerts.append("Low source coverage threshold triggered.")

        # 2. Grounding Heuristics
        raw_context = " ".join(b["content"].lower() for b in output.fetched_blocks)
        resp_words = set(output.generated_response.lower().split())
        overlap = len(resp_words & set(raw_context.split())) / max(len(resp_words), 1)
        grounded = min(1.0, overlap * 3)

        if "insufficient information" in output.generated_response.lower():
            grounded = 0.5
            alerts.append("Fallback string emitted by language model.")

        # 3. Target Alignment
        q_words = set(output.strategy.user_prompt.lower().split()) - {"what", "how", "why", "is", "the", "of"}
        alignment = min(1.0, (len(q_words & resp_words) / max(len(q_words), 1)) * 2)
        if alignment < 0.3: alerts.append("Query intent alignment falls outside confidence intervals.")
        if output.duration_seconds > 5.0: alerts.append("Inference timeout latency warning.")

        compliance = grounded >= 0.4 and alignment >= 0.3 and not any("coverage" in a for a in alerts)

        return AuditReport(
            user_prompt=output.user_prompt, generated_response=output.generated_response,
            grounded_score=round(grounded, 3), alignment_score=round(alignment, 3),
            source_coverage_ratio=round(cov, 3), passed_compliance=compliance, system_alerts=alerts
        )