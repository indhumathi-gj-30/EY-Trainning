import time
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
lexical_engine = BM25Retriever.from_documents(chunks)
lexical_engine.k = 4

# Blend semantic vector stream with literal matching
blended_search_engine = EnsembleRetriever(
    retrievers=[retriever, lexical_engine], weights=[0.6, 0.4]
)

# Verification execution for primary retrieval stage
initial_fetch_payload = blended_search_engine.invoke(
    "Apple net income fiscal 2023"
)
print(f"Total raw segments collected: {len(initial_fetch_payload)}")
for doc_chunk in initial_fetch_payload:
    print(
        f" > ID: {doc_chunk.metadata['source']} | Content snippet: {doc_chunk.page_content[:80]}..."
    )
print("-> Stage 1-3 baseline configurations successfully established.")



print("-> Initializing Neural Re-ranking Matrix (ms-marco-MiniLM)...")
deep_encoder_unit = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
refinement_processor = CrossEncoderReranker(model=deep_encoder_unit, top_n=4)

# Layer the neural ranker over the combined retrieval baseline
optimized_search_pipeline = ContextualCompressionRetriever(
    base_compressor=refinement_processor,
    base_retriever=blended_search_engine,
)

# Verification execution for refined retrieval stage
refined_fetch_payload = optimized_search_pipeline.invoke(
    "Apple net income fiscal 2023"
)
print(f"-> Segment optimization complete. Selected: {len(refined_fetch_payload)} nodes")
for doc_chunk in refined_fetch_payload:
    confidence_metric = doc_chunk.metadata.get("relevance_score")
    print(
        f" > ID: {doc_chunk.metadata['source']} (Confidence: {confidence_metric:.3f}) | {doc_chunk.page_content[:80]}..."
    )


def assemble_execution_graph(target_retrieval_layer):
    """Dynamically binds context components into standard generation workflow."""
    return (
        {
            "context": target_retrieval_layer | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# Strategy register for system benchmarking
architectures_to_test = {
    "dense": retriever,
    "hybrid": blended_search_engine,
    "hybrid_rerank": optimized_search_pipeline,
}

performance_registry = {}

for architecture_id, retrieval_component in architectures_to_test.items():
    print(f"\n[Processing Profile]: {architecture_id}")
    processing_graph = assemble_execution_graph(retrieval_component)

    generated_outputs, execution_delays = [], []

    # Execute and profile raw runtime latency
    for target_query in TEST_QUERIES:
        start_timestamp = time.time()
        model_response = processing_graph.invoke(target_query)
        execution_delays.append(time.time() - start_timestamp)
        generated_outputs.append(model_response)

    # Collect matching source text blocks for data evaluation
    extracted_contexts = [
        [node.page_content for node in retrieval_component.invoke(target_query)]
        for target_query in TEST_QUERIES
    ]

    # Map output tracking object to verification framework format
    evaluation_dataset = Dataset.from_dict(
        {
            "question": TEST_QUERIES,
            "answer": generated_outputs,
            "contexts": extracted_contexts,
            "ground_truth": GROUND_TRUTHS,
        }
    )

    print(f"-> Initiating automated validation matrix for {architecture_id}...")
    computed_metrics = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=llm,
        embeddings=az_embeddings,
    )

    performance_registry[architecture_id] = {
        "mean_latency": sum(execution_delays) / len(execution_delays),
        "data_frame": computed_metrics.to_pandas(),
    }
    print(f"   Mean response delay: {performance_registry[architecture_id]['mean_latency']:.2f}s")
    print(computed_metrics)

print("\n-> All comparative search evaluations successfully aggregated.")

matrix_summary_collection = []
for architecture_id, profile_data in performance_registry.items():
    metrics_df = profile_data["data_frame"]
    matrix_summary_collection.append(
        {
            "strategy": architecture_id,
            "avg_latency_s": round(profile_data["mean_latency"], 2),
            "faithfulness": round(metrics_df["faithfulness"].mean(), 3),
            "answer_relevancy": round(metrics_df["answer_relevancy"].mean(), 3),
            "context_recall": round(metrics_df["context_recall"].mean(), 3),
            "context_precision": round(metrics_df["context_precision"].mean(), 3),
        }
    )

analytical_dataframe = pd.DataFrame(matrix_summary_collection)
print("📊 Structural Performance Metrics Output:")
print(analytical_dataframe.to_string(index=False))

acceptable_threshold = 0.88
successful_architectures = analytical_dataframe[
    analytical_dataframe["faithfulness"] > acceptable_threshold
]["strategy"].tolist()
print(
    f"\n🎯 Target threshold condition (> {acceptable_threshold}) satisfied by: {successful_architectures or 'None (System adaptation or fine-tuning required)'}"
)

visual_canvas, axis_subplot = plt.subplots(figsize=(7, 5))
palette_mapping = {
    "dense": "#4C72B0",
    "hybrid": "#DD8452",
    "hybrid_rerank": "#55A868",
}

for _, record in analytical_dataframe.iterrows():
    axis_subplot.scatter(
        record["avg_latency_s"],
        record["faithfulness"],
        s=160,
        color=palette_mapping.get(record["strategy"], "gray"),
        label=record["strategy"],
        zorder=3,
    )
    axis_subplot.annotate(
        record["strategy"],
        (record["avg_latency_s"], record["faithfulness"]),
        textcoords="offset points",
        xytext=(8, 6),
    )

axis_subplot.axhline(
    acceptable_threshold,
    color="gray",
    linestyle="--",
    linewidth=1,
    label=f"Target ({acceptable_threshold})",
)
axis_subplot.set_xlabel("Average latency (s)")
axis_subplot.set_ylabel("Faithfulness")
axis_subplot.set_title(
    "System Latency vs. Response Faithfulness Across Search Implementations"
)
axis_subplot.legend()
plt.tight_layout()

output_graphic_filename = "faithfulness_vs_latency.png"
plt.savefig(output_graphic_filename, dpi=150)
plt.show()

print(f"-> Statistical visualization successfully exported to: {output_graphic_filename}")