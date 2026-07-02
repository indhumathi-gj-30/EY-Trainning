import os
import time
import uuid
import streamlit as st
from datetime import datetime
from pathlib import Path

from rag_engine import (
    PKLStore, RAGOrchestrator, RAGSession,
    Plan, ExecutionResult, ValidationReport
)
st.set_page_config(
    page_title="WealthIntel",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Base Overrides */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Custom Dark Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090a0f 0%, #121420 100%);
    border-right: 1px solid #222538;
}
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label { color: #94a3b8 !important; font-size: 0.78rem; }

/* Main Header */
.wealthintel-topbar {
    background: linear-gradient(135deg, #020617 0%, #111827 50%, #020617 100%);
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.wealthintel-topbar::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10b981, #3b82f6, #6366f1);
}
.wealthintel-topbar h1 { color: #f8fafc; font-size: 1.85rem; font-weight: 700; margin: 0; }
.wealthintel-topbar p  { color: #64748b; margin: 6px 0 0; font-size: 0.92rem; }

/* Live Process Tracking Pipeline */
.workflow-container {
    display: flex; align-items: center; gap: 0;
    background: #0b0f19; border: 1px solid #1e293b;
    border-radius: 10px; padding: 12px 18px; margin-bottom: 18px;
}
.workflow-node {
    flex: 1; text-align: center; padding: 8px 6px;
    background: #1e293b; border-radius: 6px; margin: 0 4px;
}
.workflow-node.active { background: #059669; border: 1px solid #10b981; }
.workflow-node.done   { background: #1e3a8a; border: 1px solid #3b82f6; }
.workflow-node.error  { background: #991b1b; border: 1px solid #ef4444; }
.workflow-node h4 { color: #f8fafc; margin: 0; font-size: 0.82rem; font-weight: 600; }
.workflow-node p  { color: #94a3b8; margin: 2px 0 0; font-size: 0.7rem; }
.workflow-arrow   { color: #334155; font-size: 1.1rem; padding: 0 4px; flex-shrink: 0; }

/* Performance Counters */
.stat-box {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 10px; padding: 16px 18px; text-align: center;
}
.stat-box h3 { color: #64748b; font-size: 0.75rem; font-weight: 500;
               text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 6px; }
.stat-box .num-value { color: #f8fafc; font-size: 1.65rem; font-weight: 700; }
.stat-box .sub-label { color: #475569; font-size: 0.72rem; }

/* Intent Tag Pill Shapes */
.tag-pill {
    display: inline-block; padding: 2px 8px; border-radius: 16px;
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
}
.tag-revenue    { background: #1e3a8a33; color: #60a5fa; border: 1px solid #2563eb; }
.tag-risk       { background: #991b1b33; color: #f87171; border: 1px solid #dc2626; }
.tag-liquidity  { background: #064e3b33; color: #34d399; border: 1px solid #059669; }
.tag-comparative{ background: #581c8733; color: #c084fc; border: 1px solid #7c3aed; }
.tag-trend      { background: #7c2d1233; color: #fbbf24; border: 1px solid #d97706; }
.tag-factual    { background: #334155;   color: #cbd5e1; border: 1px solid #475569; }

/* Output Wrapper */
.response-wrapper {
    background: #020617; border: 1px solid #1e293b;
    border-left: 4px solid #10b981;
    border-radius: 0 10px 10px 0; padding: 18px 22px;
    color: #cbd5e1; line-height: 1.65; font-size: 0.92rem;
    margin: 10px 0;
}

/* Linear Gauge Indicators */
.gauge-track { background: #334155; border-radius: 10px; height: 6px; overflow: hidden; }
.gauge-fill  { height: 6px; border-radius: 10px;
               background: linear-gradient(90deg, #10b981, #3b82f6); }

/* Reference Tags */
.doc-reference {
    display: inline-block; padding: 3px 8px; margin: 2px;
    background: #1e293b; border: 1px solid #475569;
    border-radius: 4px; font-size: 0.7rem; color: #94a3b8;
}

/* Linear Logs View */
.log-item-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; background: #0f172a;
    border: 1px solid #1e293b; border-radius: 6px; margin-bottom: 5px;
}
.log-item-row:hover { border-color: #475569; }

/* Context Scroll Pane */
.scroll-panel { max-height: 240px; overflow-y: auto;
                border: 1px solid #1e293b; border-radius: 6px; padding: 6px; }
.block-card { background: #111827; border-radius: 5px; padding: 8px 10px;
              margin-bottom: 5px; font-size: 0.78rem; color: #9ca3af; }
.block-card .block-origin { color: #10b981; font-weight: 600; font-size: 0.7rem; }

/* Flag Elements */
.audit-pass { background: #064e3b; color: #34d399; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
.audit-fail { background: #7f1d1d; color: #f87171; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }

/* Alert Indicator Boxes */
.alert-box {
    background: #7c2d1222; border: 1px solid #c2410c;
    border-radius: 4px; padding: 6px 10px; margin: 3px 0;
    color: #f59e0b; font-size: 0.8rem;
}

/* Streamlit Native Modifications */
.stButton>button {
    background: linear-gradient(135deg, #059669, #2563eb);
    color: white; border: none; border-radius: 6px;
    font-weight: 600; padding: 8px 20px;
    transition: opacity .15s;
}
.stButton>button:hover { opacity: .9; color: white; }
div[data-testid="stTextInput"] input {
    background: #020617; color: #f8fafc;
    border: 1px solid #475569; border-radius: 6px;
}
div[data-testid="stTextInput"] input:focus { border-color: #10b981; box-shadow: none; }
.stTabs [data-baseweb="tab"] { color: #475569; }
.stTabs [aria-selected="true"] { color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Embedded Mock Assets ─────────────────────────────────────────────────────

SAMPLE_DOCS = [
    {
        "source": "Apple_10K_2023_Risk",
        "text": """RISK FACTORS
Apple's operations and financial results are subject to various risks and uncertainties.
Global and regional economic conditions, including conditions resulting from financial and credit market fluctuations,
can adversely affect demand for Apple's products and services.
Apple faces intense competition in all of its business areas from well-established companies with significant
resources, as well as from new market entrants.
The Company's fiscal year 2023 revenue was $383.3 billion, compared to $394.3 billion in fiscal 2022,
a decrease of approximately 2.8 percent.
The Company's net income for fiscal 2023 was $97.0 billion, or $6.13 diluted earnings per share,
compared to $99.8 billion, or $6.11 diluted earnings per share, in fiscal 2022.
Apple's gross margin percentage was 44.1% in fiscal 2023, compared to 43.3% in fiscal 2022.
Services revenue reached an all-time high of $85.2 billion in fiscal 2023, up 9 percent year over year."""
    },
    {
        "source": "Apple_10K_2023_Products",
        "text": """PRODUCTS AND SERVICES
Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories.
iPhone is Apple's line of smartphones based on its iOS operating system.
iPhone net sales were $200.6 billion in fiscal 2023, representing approximately 52% of total revenue.
Mac net sales were $29.4 billion in fiscal 2023, down from $40.2 billion in fiscal 2022.
iPad net sales were $28.3 billion in fiscal 2023.
Wearables, Home and Accessories net sales were $39.8 billion in fiscal 2023.
Apple's Services segment includes advertising, AppleCare, cloud, digital content, payment and other services.
The App Store, Apple Music, Apple TV+, Apple Arcade, iCloud and Apple Pay are key Services offerings.
The Company had approximately 2.2 billion active devices at the end of fiscal year 2023."""
    },
    {
        "source": "Apple_10K_2023_Liquidity",
        "text": """LIQUIDITY AND CAPITAL RESOURCES
The Company believes its existing balances of cash, cash equivalents and unrestricted marketable securities,
together with cash generated by operations, will be sufficient to satisfy its expected cash needs.
Cash and cash equivalents as of September 30, 2023 were $29.965 billion.
Total marketable securities were $100.544 billion, consisting of current marketable securities of $31.590 billion
and non-current marketable securities of $100.544 billion.
During fiscal 2023, the Company returned over $77 billion to shareholders,
including $15.1 billion in dividends and dividend equivalents and $62.2 billion through repurchases of 471 million shares.
Capital expenditures were $10.959 billion in fiscal 2023.
The Company's long-term debt as of September 30, 2023 was $95.281 billion."""
    },
]

SAMPLE_QUERIES = [
    "What was Apple's total revenue in fiscal year 2023?",
    "How much cash did Apple have at the end of fiscal 2023?",
    "What percentage of Apple's revenue came from iPhone in 2023?",
    "How much did Apple return to shareholders in fiscal 2023?",
    "What is Apple's gross margin for fiscal 2023?",
    "What risks does Apple face in its business?",
    "Compare iPhone vs Mac revenue for fiscal 2023",
]


@st.cache_resource
def setup_storage_engine():
    return PKLStore("./rag_store")


def manage_user_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def select_hex_by_score(val: float) -> str:
    if val >= 0.7:
        return "#10b981"
    elif val >= 0.4:
        return "#d97706"
    return "#dc2626"


def display_pipeline_progress(current_stage: str):
    """current_stage: idle | planning | executing | validating | done | error"""
    pipeline_nodes = [
        ("🗺️", "Strategy", "Intent Evaluation"),
        ("⚡", "Execution", "Lookup & Synthesis"),
        ("✅", "Audit",    "Safety Validation"),
    ]
    progress_map = {"idle": -1, "planning": 0, "executing": 1, "validating": 2, "done": 3, "error": -2}
    highlight_idx = progress_map.get(current_stage, -1)

    dom_string = '<div class="workflow-container">'
    for idx, (emoji, title, desc) in enumerate(pipeline_nodes):
        state_class = "done" if idx < highlight_idx else ("active" if idx == highlight_idx else "")
        if current_stage == "error" and idx == highlight_idx:
            state_class = "error"
        dom_string += f'<div class="workflow-node {state_class}"><h4>{emoji} {title}</h4><p>{desc}</p></div>'
        if idx < 2:
            dom_string += '<span class="workflow-arrow">→</span>'
    dom_string += "</div>"
    st.markdown(dom_string, unsafe_allow_html=True)


def parse_intent_css(intent_type: str):
    return f'<span class="tag-pill tag-{intent_type}">{intent_type}</span>'


def build_linear_gauge(title: str, ratio: float):
    target_color = select_hex_by_score(ratio)
    percentage = int(ratio * 100)
    st.markdown(f"""
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;margin-bottom:2px">
        <span style="color:#64748b;font-size:0.78rem">{title}</span>
        <span style="color:{target_color};font-weight:600;font-size:0.78rem">{percentage}%</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{percentage}%;background:{target_color}"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─── Data Embedding Pipeline ──────────────────────────────────────────────────

def compile_vector_cache(storage_handler: PKLStore, system_config: dict):
    """Processes datasets via recursive splits, embeds into FAISS and preserves state."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    size_limit   = system_config.get("chunk_size", 512)
    step_overlap = system_config.get("chunk_overlap", 64)

    text_parser = RecursiveCharacterTextSplitter(
        chunk_size=size_limit,
        chunk_overlap=step_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    wrapped_docs = [Document(page_content=d["text"], metadata={"source": d["source"]})
                    for d in SAMPLE_DOCS]
    parsed_blocks = text_parser.split_documents(wrapped_docs)

    embeddings_client = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    faiss_engine = FAISS.from_documents(parsed_blocks, embeddings_client)

    meta_payload = {
        "chunk_size":    size_limit,
        "chunk_overlap": step_overlap,
        "n_chunks":      len(parsed_blocks),
        "n_vectors":     faiss_engine.index.ntotal,
        "dimension":     faiss_engine.index.d,
        "sources":       [d["source"] for d in SAMPLE_DOCS],
        "built_at":      datetime.now().isoformat(),
    }
    storage_handler.save_index(faiss_engine, parsed_blocks, meta_payload)
    storage_handler.save_config(system_config)
    return faiss_engine, parsed_blocks, embeddings_client, meta_payload


@st.cache_resource
def prime_embeddings_client():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def initialize_language_model(system_config: dict):
    target_provider = system_config.get("llm_provider", "azure")
    if target_provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=system_config["azure_endpoint"],
            azure_deployment=system_config["azure_deployment"],
            openai_api_version=system_config.get("azure_api_version", "2024-06-01"),
            openai_api_key=system_config["azure_api_key"],
            temperature=0,
            max_tokens=512,
        )
    elif target_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=system_config.get("openai_model", "gpt-4o"),
            api_key=system_config["openai_api_key"],
            temperature=0,
            max_tokens=512,
        )
    else:
        raise ValueError(f"Provider option unrecognized: {target_provider}")


# ─── Sidebar View ─────────────────────────────────────────────────────────────

def build_sidebar_controls(storage_handler: PKLStore) -> dict:
    with st.sidebar:
        st.markdown("## 🤖 WealthIntel")
        st.markdown("<hr style='border-color:#222538'>", unsafe_allow_html=True)

        # Provider Config Block
        st.markdown("### 🔑 Gateway Credentials")
        active_provider = st.selectbox("API Engine", ["Azure OpenAI", "OpenAI"], key="active_provider")

        runtime_settings = {}
        if active_provider == "Azure OpenAI":
            runtime_settings["llm_provider"]      = "azure"
            runtime_settings["azure_endpoint"]    = st.text_input("Endpoint Target URL", type="password",
                                                                 placeholder="https://…openai.azure.com/")
            runtime_settings["azure_api_key"]     = st.text_input("Secret Token Key", type="password")
            runtime_settings["azure_deployment"]  = st.text_input("Deployment Identifier", value="gpt-4o")
            runtime_settings["azure_api_version"] = "2024-06-01"
        else:
            runtime_settings["llm_provider"]   = "openai"
            runtime_settings["openai_api_key"] = st.text_input("OpenAI User Key", type="password")
            runtime_settings["openai_model"]   = st.selectbox("Model Selector", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])

        st.markdown("<hr style='border-color:#222538'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Parser Boundaries")

        runtime_settings["chunk_size"]         = st.select_slider("Partition Size", [256, 512, 1024], value=512)
        runtime_settings["chunk_overlap"]      = st.slider("Step Overlap", 32, 128, 64, 16)
        runtime_settings["retrieval_k"]        = st.slider("Target Node Count (k)", 2, 8, 4)
        runtime_settings["retrieval_strategy"] = st.radio("Strategy Pattern", ["dense", "hybrid"], horizontal=True)

        st.markdown("<hr style='border-color:#222538'>", unsafe_allow_html=True)

        # Validation Checks & Setup Calls
        cached_index_found = storage_handler.index_exists()
        action_title = "🔄 Overwrite Cache" if cached_index_found else "🏗️ Construct Index"

        if st.button(action_title, use_container_width=True):
            with st.spinner("Compiling structural vectors…"):
                try:
                    vs, blocks, ec, meta = compile_vector_cache(storage_handler, runtime_settings)
                    st.session_state["vectorstore"]     = vs
                    st.session_state["chunks"]          = blocks
                    st.session_state["embedding_model"] = ec
                    st.session_state["index_meta"]      = meta
                    st.success(f"✅ Success — Loaded {meta['n_vectors']} records")
                except Exception as ex:
                    st.error(f"Setup execution failed: {ex}")

        # Metrics Display Panel
        if cached_index_found:
            _, _, meta = storage_handler.load_index()
            if meta:
                st.markdown(f"""
                <div style="background:#090a0f;border:1px solid #1e293b;border-radius:6px;padding:8px 10px;margin-top:6px">
                  <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em">Dataset Matrix</div>
                  <div style="color:#cbd5e1;font-size:0.78rem;margin-top:3px">
                    {meta.get('n_vectors','?')} shapes · {meta.get('dimension','?')}D<br>
                    Blocks: {meta.get('n_chunks','?')} · Bounds: {meta.get('chunk_size','?')}<br>
                    Compiled: {meta.get('built_at','?')[:10]}
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#222538'>", unsafe_allow_html=True)

        # File Object Metrics
        st.markdown("### 💾 Storage Status")
        cache_metrics = storage_handler.store_stats()
        for metric_key, metric_val in cache_metrics.items():
            st.markdown(f"<div style='color:#64748b;font-size:0.75rem'>{metric_key}: <b style='color:#cbd5e1'>{metric_val}</b></div>",
                        unsafe_allow_html=True)

        if st.button("🗑️ Wipe Query Logs", use_container_width=True):
            storage_handler.clear_history()
            st.success("Internal pipeline histories wiped.")

    return runtime_settings


# ─── Primary Interface Architecture ───────────────────────────────────────────

def main():
    storage_handler = setup_storage_engine()
    runtime_settings = build_sidebar_controls(storage_handler)
    active_session_id = manage_user_session()

    # Core Banner Section
    st.markdown("""
    <div class="wealthintel-topbar">
      <h1>🤖 WealthIntel — Financial RAG System</h1>
      <p>Strategy → Execution → Audit · High Performance FAISS · Persistent File Storage</p>
    </div>""", unsafe_allow_html=True)

    # Navigation Controls
    query_tab, log_tab, session_tab, document_tab = st.tabs([
        "🔍 Evaluation Panel", "📋 Run Logs", "🗂️ Session Ledger", "📄 Source Inventories"
    ])

    # ── TAB: PERFORMANCE / EVALUATION ─────────────────────────────────────────
    with query_tab:
        active_pipeline_stage = st.session_state.get("pipeline_stage", "idle")
        display_pipeline_progress(active_pipeline_stage)

        # Primary input bar
        input_col, submission_col = st.columns([5, 1])
        with input_col:
            user_prompt = st.text_input(
                "Analyst Input Target",
                placeholder="Submit your explicit financial queries here...",
                label_visibility="collapsed",
                key="query_input",
            )
        with submission_col:
            trigger_execution = st.button("▶ Run", use_container_width=True)

        # Shortcut Elements
        st.markdown("<div style='margin-bottom:12px'>", unsafe_allow_html=True)
        shortcut_layout = st.columns(len(SAMPLE_QUERIES))
        for index, item_text in enumerate(SAMPLE_QUERIES):
            with shortcut_layout[index]:
                truncated_text = item_text[:28] + "…" if len(item_text) > 28 else item_text
                if st.button(truncated_text, key=f"shortcut_{index}", use_container_width=True, help=item_text):
                    st.session_state["query_input"] = item_text
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Execution Lifecycle Handler
        if trigger_execution and user_prompt:
            credentials_verified = (
                (runtime_settings.get("azure_api_key") and runtime_settings.get("azure_endpoint")) or
                runtime_settings.get("openai_api_key")
            )
            if not credentials_verified:
                st.error("🔑 Gateway token requirements unfulfilled. Review sidebar parameters.")
                st.stop()

            if "vectorstore" not in st.session_state:
                if storage_handler.index_exists():
                    with st.spinner("Retrieving stored models..."):
                        vs, blocks, meta = storage_handler.load_index()
                        ec = prime_embeddings_client()
                        st.session_state.update({
                            "vectorstore": vs,
                            "chunks": blocks,
                            "embedding_model": ec,
                            "index_meta": meta,
                        })
                else:
                    with st.spinner("Initializing structural vectors (Cold Start)..."):
                        vs, blocks, ec, meta = compile_vector_cache(storage_handler, runtime_settings)
                        st.session_state.update({
                            "vectorstore": vs,
                            "chunks": blocks,
                            "embedding_model": ec,
                            "index_meta": meta,
                        })

            try:
                llm_instance = initialize_language_model(runtime_settings)
            except Exception as ex:
                st.error(f"Failed to bring up model client: {ex}")
                st.stop()

            pipeline_orchestrator = RAGOrchestrator(storage_handler)
            pipeline_orchestrator.load_or_init_session(active_session_id)
            pipeline_orchestrator.setup_executor(
                st.session_state["vectorstore"],
                st.session_state["embedding_model"],
                llm_instance
            )

            # Cycle State Transition
            st.session_state["pipeline_stage"] = "planning"
            st.rerun()

        current_internal_state = st.session_state.get("pipeline_stage", "idle")

        if current_internal_state == "planning" and user_prompt:
            from rag_engine import Planner
            strategy_planner = Planner()
            execution_plan = strategy_planner.plan(user_prompt, runtime_settings)
            st.session_state["current_plan"] = execution_plan
            st.session_state["pipeline_stage"] = "executing"
            display_pipeline_progress("planning")
            st.rerun()

        elif current_internal_state == "executing" and "current_plan" in st.session_state:
            display_pipeline_progress("executing")
            execution_plan = st.session_state["current_plan"]

            if "vectorstore" not in st.session_state:
                st.session_state["pipeline_stage"] = "idle"
                st.rerun()

            try:
                credentials_verified = (
                    (runtime_settings.get("azure_api_key") and runtime_settings.get("azure_endpoint")) or
                    runtime_settings.get("openai_api_key")
                )
                if not credentials_verified:
                    st.error("Missing credentials during active execution pass")
                    st.session_state["pipeline_stage"] = "idle"
                    st.stop()

                llm_instance = initialize_language_model(runtime_settings)
                from rag_engine import Executor, Validator
                task_executor = Executor(st.session_state["vectorstore"],
                                         st.session_state["embedding_model"], llm_instance)
                
                with st.spinner("⚡ Running data synthesizer..."):
                    runtime_output = task_executor.execute(execution_plan)
                st.session_state["current_result"] = runtime_output
                st.session_state["pipeline_stage"] = "validating"
                st.rerun()

            except Exception as ex:
                st.error(f"Active run layer failed: {ex}")
                st.session_state["pipeline_stage"] = "idle"

        elif current_internal_state == "validating" and "current_result" in st.session_state:
            display_pipeline_progress("validating")
            runtime_output = st.session_state["current_result"]
            execution_plan = st.session_state["current_plan"]

            try:
                credentials_verified = (
                    (runtime_settings.get("azure_api_key") and runtime_settings.get("azure_endpoint")) or
                    runtime_settings.get("openai_api_key")
                )
                llm_instance = initialize_language_model(runtime_settings) if credentials_verified else None
                from rag_engine import Validator
                audit_validator = Validator(llm_instance)
                audit_report = audit_validator.validate(runtime_output)

                # Append session records
                pipeline_orchestrator = RAGOrchestrator(storage_handler)
                pipeline_orchestrator.load_or_init_session(active_session_id)
                pipeline_orchestrator._session.queries.append(vars(execution_plan))
                pipeline_orchestrator._session.executions.append({
                    "query": runtime_output.query, "answer": runtime_output.answer,
                    "latency": runtime_output.latency_s, "chunks": runtime_output.retrieved_chunks,
                    "tokens": runtime_output.token_count, "timestamp": runtime_output.timestamp,
                })
                from dataclasses import asdict
                pipeline_orchestrator._session.validations.append(asdict(audit_report))
                storage_handler.save_session(pipeline_orchestrator._session)
                storage_handler.append_history({
                    "query": user_prompt, "intent": execution_plan.intent,
                    "answer": runtime_output.answer, "latency_s": runtime_output.latency_s,
                    "faithfulness": audit_report.faithfulness_score,
                    "relevance": audit_report.relevance_score,
                    "passed": audit_report.passed, "timestamp": runtime_output.timestamp,
                })

                st.session_state["current_report"] = audit_report
                st.session_state["pipeline_stage"]  = "done"
                st.rerun()

            except Exception as ex:
                st.error(f"Audit generation exception: {ex}")
                st.session_state["pipeline_stage"] = "idle"

        elif current_internal_state == "done":
            display_pipeline_progress("done")
            execution_plan = st.session_state.get("current_plan")
            runtime_output = st.session_state.get("current_result")
            audit_report   = st.session_state.get("current_report")

            if execution_plan and runtime_output and audit_report:
                present_computed_elements(execution_plan, runtime_output, audit_report)

                if st.button(" Initialize Fresh Prompt"):
                    for state_key in ["pipeline_stage", "current_plan", "current_result", "current_report"]:
                        st.session_state.pop(state_key, None)
                    st.rerun()

    # ── TAB: ARCHIVED HISTORY ─────────────────────────────────────────────────
    with log_tab:
        full_history_logs = storage_handler.load_history()
        st.markdown(f"### Historical Inquiries &nbsp; <span style='color:#64748b;font-size:0.82rem'>({len(full_history_logs)} entries)</span>",
                    unsafe_allow_html=True)

        if not full_history_logs:
            st.info("No recorded runs available. Launch an inquiry pass.")
        else:
            grid_col1, grid_col2, grid_col3, grid_col4 = st.columns(4)
            computed_latency = sum(item["latency_s"] for item in full_history_logs) / len(full_history_logs)
            computed_accuracy = sum(item.get("faithfulness", 0) for item in full_history_logs) / len(full_history_logs)
            overall_pass_rate = sum(1 for item in full_history_logs if item.get("passed")) / len(full_history_logs)

            for target_col, summary_title, numeric_value, suffix in [
                (grid_col1, "Total Operations", len(full_history_logs), ""),
                (grid_col2, "Mean Latency",     f"{computed_latency:.2f}", "s"),
                (grid_col3, "Context Matching", f"{computed_accuracy:.0%}", ""),
                (grid_col4, "Audit Compliance", f"{overall_pass_rate:.0%}", ""),
            ]:
                with target_col:
                    st.markdown(f"""<div class="stat-box">
                      <h3>{summary_title}</h3>
                      <div class="num-value">{numeric_value}<span class="sub-label">{suffix}</span></div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for log_node in reversed(full_history_logs[-20:]):
                badge_dom = parse_intent_css(log_node.get("intent", "factual"))
                status_flag_dom = f'<span class="audit-pass">PASS</span>' if log_node.get("passed") else f'<span class="audit-fail">FAIL</span>'
                st.markdown(f"""
                <div class="log-item-row">
                  <div style="flex:1">
                    <div style="color:#cbd5e1;font-size:0.82rem;font-weight:500">{log_node['query'][:80]}</div>
                    <div style="color:#475569;font-size:0.7rem;margin-top:2px">{log_node.get('timestamp','')[:19]}</div>
                  </div>
                  <div>{badge_dom}</div>
                  <div style="color:#94a3b8;font-size:0.78rem;min-width:55px;text-align:right">{log_node.get('latency_s',0):.2f}s</div>
                  <div style="min-width:45px;text-align:right">{status_flag_dom}</div>
                </div>""", unsafe_allow_html=True)

    # ── TAB: PERSISTENT SESSION TRACKER ───────────────────────────────────────
    with session_tab:
        st.markdown("### Ledger Scope Target")
        active_ledger_instances = storage_handler.load_all_sessions()
        if not active_ledger_instances:
            st.info("Cache storage handles no active sessions.")
        else:
            for instance_id, container in active_ledger_instances.items():
                target_queries = container.queries if hasattr(container, 'queries') else container.get('queries', [])
                with st.expander(f"📁 Session Hash: {instance_id} — {len(target_queries)} elements · {container.created_at[:10]}"):
                    execution_entries = container.executions if hasattr(container, "executions") else container.get("executions", [])
                    audit_entries     = container.validations if hasattr(container, "validations") else container.get("validations", [])
                    st.markdown(f"**Synthesized Iterations:** {len(execution_entries)} &nbsp; **Audit Reports:** {len(audit_entries)}")
                    if execution_entries:
                        for individual_exec in execution_entries[-3:]:
                            st.markdown(f"- `{individual_exec.get('query','')[:55]}` · {individual_exec.get('latency',0):.2f}s")

    # ── TAB: STATIC DATA VIEWS ────────────────────────────────────────────────
    with document_tab:
        st.markdown("### 📄 Active File Registry")
        for asset_doc in SAMPLE_DOCS:
            with st.expander(f"📑 Document Context: {asset_doc['source']}"):
                st.code(asset_doc["text"], language="text")

        if "index_meta" in st.session_state:
            meta = st.session_state["index_meta"]
            st.markdown("### 📊 Index Distribution Summary")
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            for specific_col, text_header, numeric_metric in [
                (sub_col1, "Vector Dimension Count", meta.get("n_vectors", "?")),
                (sub_col2, "Vector Coordinates Shape", meta.get("dimension", "?")),
                (sub_col3, "Total Base File Segments", meta.get("n_chunks", "?")),
            ]:
                with specific_col:
                    st.markdown(f"""<div class="stat-box">
                      <h3>{text_header}</h3>
                      <div class="num-value">{numeric_metric}</div>
                    </div>""", unsafe_allow_html=True)


# ─── Downstream Processing Visualizers ────────────────────────────────────────

def present_computed_elements(execution_plan: Plan, runtime_output: ExecutionResult, audit_report: ValidationReport):
    st.markdown("<br>", unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    for target_col, visual_label, raw_numeric, measuring_unit in [
        (metric_col1, "Latency Metrics",   f"{runtime_output.latency_s:.2f}", "s"),
        (metric_col2, "Extracted Nodes",   len(runtime_output.retrieved_chunks), ""),
        (metric_col3, "Context Overlap",   f"{audit_report.faithfulness_score:.0%}", ""),
        (metric_col4, "Keyword Relevance", f"{audit_report.relevance_score:.0%}", ""),
        (metric_col5, "Estimated Tokens",  runtime_output.token_count, ""),
    ]:
        with target_col:
            st.markdown(f"""<div class="stat-box">
              <h3>{visual_label}</h3>
              <div class="num-value">{raw_numeric}<span class="sub-label"> {measuring_unit}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    split_left, split_right = st.columns([3, 2])

    with split_left:
        badge_dom_string = parse_intent_css(execution_plan.intent)
        st.markdown(f"""
        <div style="margin-bottom:10px">
          <span style="color:#64748b;font-size:0.75rem">TARGET PATTERN:</span>&nbsp;&nbsp;{badge_dom_string}&nbsp;&nbsp;
          <span style="color:#64748b;font-size:0.75rem">LIMIT={execution_plan.retrieval_k}</span>&nbsp;&nbsp;
          <span style="color:#64748b;font-size:0.75rem">BOUNDS={execution_plan.chunk_size}</span>&nbsp;&nbsp;
          <span style="color:#64748b;font-size:0.75rem">FLOW={execution_plan.strategy}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 💬 Synthesized System Response")
        st.markdown(f'<div class="response-wrapper">{runtime_output.answer}</div>', unsafe_allow_html=True)

        st.markdown("#### 📎 Data Source Mappings")
        reference_chips = "".join(f'<span class="doc-reference">{chunk_node["source"]}</span>'
                                  for chunk_node in runtime_output.retrieved_chunks)
        st.markdown(reference_chips, unsafe_allow_html=True)

        with st.expander("🔍 Examine Retained File Blocks"):
            st.markdown('<div class="scroll-panel">', unsafe_allow_html=True)
            for idx, chunk_node in enumerate(runtime_output.retrieved_chunks):
                st.markdown(f"""<div class="block-card">
                  <div class="block-origin">Node #{idx+1} · Context Source: {chunk_node['source']}</div>
                  <div style="margin-top:3px">{chunk_node['content'][:300]}{'…' if len(chunk_node['content'])>300 else ''}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with split_right:
        status_flag_dom = '<span class="audit-pass">✅ AUDIT PASSED</span>' if audit_report.passed else '<span class="audit-fail">❌ AUDIT FAILED</span>'
        st.markdown(f"#### Security Verification &nbsp; {status_flag_dom}", unsafe_allow_html=True)

        build_linear_gauge("Context Overlap",   audit_report.faithfulness_score)
        build_linear_gauge("Keyword Relevance",  audit_report.relevance_score)
        build_linear_gauge("Source Attribution", audit_report.source_coverage)

        if audit_report.warnings:
            st.markdown("**⚠️ Flagged Alerts**")
            for alert_text in audit_report.warnings:
                st.markdown(f'<div class="alert-box">⚠️ {alert_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#064e3b22;border:1px solid #064e3b;border-radius:4px;padding:6px 10px;color:#34d399;font-size:0.78rem">✅ System reports zero structural flags</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>**Structural Plan Output**")
        st.json({
            "intent":           execution_plan.intent,
            "strategy":         execution_plan.strategy,
            "retrieval_k":      execution_plan.retrieval_k,
            "chunk_size":       execution_plan.chunk_size,
            "required_sources": execution_plan.required_sources,
        })


if __name__ == "__main__":
    main()