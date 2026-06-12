if not COHERE_API_KEY:
    print("[WARNING] COHERE_API_KEY not configured. Skipping evaluation pipeline.")
else:
    import cohere

    class SearchQualityEvaluator:
        """
        An enterprise-grade evaluator to benchmark vector search against 
        Cohere's two-stage re-ranking architecture using Mean Reciprocal Rank (MRR).
        """
        def __init__(self, api_key: str):
            self.client = cohere.Client(api_key)
            self.test_cases = {
                "How does FAISS handle approximate nearest neighbour search?": "FAISS",
                "What is the difference between BM25 and dense retrieval?": "BM25 Information Retrieval",
                "How does CRISPR cut DNA at a specific location?": "CRISPR Gene Editing",
                "What triggered the COVID-19 pandemic?": "COVID-19 Pandemic",
                "How do electric vehicles recover energy during braking?": "Electric Vehicles"
            }

        def score_reciprocal_rank(self, docs, expected_label: str) -> float:
            """Determines the reciprocal rank by analyzing document metadata."""
            total_docs = len(docs)
            for idx in range(total_docs):
                current_doc = docs[idx]
                title_field = current_doc.metadata.get("title", "")
                if expected_label.lower() in title_field.lower():
                    return 1.0 / (idx + 1)
            return 0.0

        def execute_pipeline(self, vector_store, k_pull=10, k_truncate=10):
            """Runs retrieval, reranking, and logs comparative metrics."""
            before_scores = []
            after_scores = []

            for query_str, target_title in self.test_cases.items():
              
                base_documents = vector_store.similarity_search(query_str, k=k_pull)
                text_corpus = [d.page_content for d in base_documents]

                rerank_payload = self.client.rerank(
                    model="rerank-english-v3.0",
                    query=query_str,
                    documents=text_corpus,
                    top_n=k_truncate
                )
                optimized_documents = []
                for element in rerank_payload.results:
                    source_index = element.index
                    optimized_documents.append(base_documents[source_index])
                before_scores.append(self.score_reciprocal_rank(base_documents, target_title))
                after_scores.append(self.score_reciprocal_rank(optimized_documents, target_title))

            return before_scores, after_scores

    evaluator_engine = SearchQualityEvaluator(api_key=COHERE_API_KEY)
    raw_mrr, refined_mrr = evaluator_engine.execute_pipeline(vector_store=faiss_store)

    mrr_baseline_avg = np.mean(raw_mrr)
    mrr_optimized_avg = np.mean(refined_mrr)
    net_variance = (mrr_optimized_avg - mrr_baseline_avg) * 100

    print("--- PIPELINE PERFORMANCE METRICS ---")
    print(f"Initial Semantic Match (MRR@10) : {mrr_baseline_avg:.4f}")
    print(f"Reranked Multi-Stage (MRR@10)   : {mrr_optimized_avg:.4f}")
    print(f"Net Variance Captured           : {net_variance:+.2f}% Change")
    print("------------------------------------")
    fig = plt.figure(figsize=(6, 5))
    plotting_axis = fig.add_subplot(111)
    
    metrics_dataset = [mrr_baseline_avg, mrr_optimized_avg]
    labels_list = ['Initial FAISS Pull', 'Post Cohere Filter']
    visual_theme_colors = ['#4A5568', '#3182CE'] 

    bars_rendered = plotting_axis.bar(
        labels_list, 
        metrics_dataset, 
        color=visual_theme_colors, 
        width=0.45,
        zorder=3
    )

    plotting_axis.set_title("Retrieval Precision Delta (MRR Analysis)", fontsize=11, pad=20, weight='semibold')
    plotting_axis.set_ylabel("Score (0.0 - 1.0)", fontsize=10)
    plotting_axis.set_ylim(0, 1.2)
    plotting_axis.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)


    for single_bar in bars_rendered:
        node_height = single_bar.get_height()
        plotting_axis.text(
            single_bar.get_x() + single_bar.get_width()/2.0,
            node_height + 0.03,
            f'{node_height:.4f}',
            ha='center', 
            va='bottom', 
            fontsize=9,
            weight='bold'
        )

    plotting_axis.spines['top'].set_visible(False)
    plotting_axis.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("search_metrics_variance.png", dpi=250)
    plt.show()

    print("[SUCCESS] Comparative visualization saved as 'search_metrics_variance.png'")