import re
import math
from rank_bm25 import BM25Okapi
# -------------------------------------------------------
# 1. Company Information Repository (Knowledge Base)
# -------------------------------------------------------
corp_knowledge_base = [
    {
        "id": 0,
        "question": "What is the leave policy?",
        "answer": "Employees can take 20 paid leaves per year. Sick leave and casual leave are included in this policy."
    },
    {
        "id": 1,
        "question": "How do I apply for work from home?",
        "answer": "You can apply for work from home through the HR portal and get approval from your manager."
    },
    {
        "id": 2,
        "question": "What are the office working hours?",
        "answer": "The standard office working hours are from 9:00 AM to 6:00 PM, Monday to Friday."
    },
    {
        "id": 3,
        "question": "How do I reset my company email password?",
        "answer": "You can reset your company email password using the self-service password reset portal or contact IT support."
    },
    {
        "id": 4,
        "question": "What is the reimbursement policy?",
        "answer": "Employees can submit travel and food reimbursement claims through the finance portal with valid bills."
    },
    {
        "id": 5,
        "question": "How can I contact HR?",
        "answer": "You can contact HR by email at hr@company.com or by visiting the HR helpdesk."
    },
    {
        "id": 6,
        "question": "What should I do if my laptop is not working?",
        "answer": "If your laptop is not working, raise a ticket with the IT support team through the service portal."
    },
    {
        "id": 7,
        "question": "Where can I find the company holiday list?",
        "answer": "The company holiday list is available in the HR portal under the holidays section."
    }
]

# -------------------------------------------------------
# 2. Validation / Ground Truth Data
# -------------------------------------------------------
test_ground_truth = [
    {"query": "leave policy", "relevant_ids": [0]},
    {"query": "work from home request", "relevant_ids": [1]},
    {"query": "office hours", "relevant_ids": [2]},
    {"query": "forgot email password", "relevant_ids": [3]},
    {"query": "travel reimbursement", "relevant_ids": [4]},
    {"query": "hr contact", "relevant_ids": [5]},
    {"query": "laptop issue", "relevant_ids": [6]},
    {"query": "company holidays", "relevant_ids": [7]}
]

# -------------------------------------------------------
# 3. Test Inputs (Sample Prompts)
# -------------------------------------------------------
eval_prompts = [
    "leave policy",
    "how to apply work from home",
    "office timing",
    "reset email password",
    "reimbursement claim",
    "contact hr",
    "laptop not working",
    "holiday list"
]

# -------------------------------------------------------
# 4. Text Preprocessing Function
# -------------------------------------------------------
def sanitize_text(raw_text):
    lowered_text = raw_text.lower()
    cleaned_text = re.sub(r"[^a-z0-9\s]", "", lowered_text)
    return cleaned_text.split()

# -------------------------------------------------------
# 5. Process Documents for BM25 Engine
# -------------------------------------------------------
corpus_docs = []
for doc_item in corp_knowledge_base:
    merged_string = doc_item["question"] + " " + doc_item["answer"]
    corpus_docs.append(merged_string)

tokenized_docs = [sanitize_text(d) for d in corpus_docs]

# Instantiate BM25 Model
search_engine = BM25Okapi(tokenized_docs)

# -------------------------------------------------------
# 6. Fetch Top-K Matches
# -------------------------------------------------------
def get_top_matches(query_text, limit_k=5):
    query_tokens = sanitize_text(query_text)
    match_scores = search_engine.get_scores(query_tokens)

    sorted_indices = sorted(range(len(match_scores)), key=lambda idx: match_scores[idx], reverse=True)

    matched_records = []
    for idx in sorted_indices[:limit_k]:
        matched_records.append({
            "id": corp_knowledge_base[idx]["id"],
            "question": corp_knowledge_base[idx]["question"],
            "answer": corp_knowledge_base[idx]["answer"],
            "score": float(match_scores[idx])
        })

    return matched_records

# -------------------------------------------------------
# 7. Extract Target IDs from Ground Truth
# -------------------------------------------------------
def fetch_target_ids(query_str):
    normalized_query = query_str.strip().lower()

    for data_node in test_ground_truth:
        if data_node["query"].strip().lower() == normalized_query:
            return data_node["relevant_ids"]

    return None

# -------------------------------------------------------
# 8. Compute Reciprocal Rank for Single Query
# -------------------------------------------------------
def compute_rr(found_ids, target_ids):
    for rank_idx, doc_id in enumerate(found_ids, start=1):
        if doc_id in target_ids:
            return 1.0 / rank_idx
    return 0.0

# -------------------------------------------------------
# 9. Compute Mean Reciprocal Rank (MRR)
# -------------------------------------------------------
def evaluate_mrr(dataset, limit_k=5):
    rr_list = []

    for entry in dataset:
        q_text = entry["query"]
        t_ids = entry["relevant_ids"]

        top_hits = get_top_matches(q_text, limit_k=limit_k)
        hit_ids = [hit["id"] for hit in top_hits]

        rr_val = compute_rr(hit_ids, t_ids)
        rr_list.append(rr_val)

    if not rr_list:
        return 0.0

    return sum(rr_list) / len(rr_list)

# -------------------------------------------------------
# 10. Discounted Cumulative Gain (DCG) Calculation
# -------------------------------------------------------
def compute_dcg(relevance_array):
    dcg_sum = 0.0
    for pos_idx, rel_score in enumerate(relevance_array):
        rank_pos = pos_idx + 1
        dcg_sum += rel_score / math.log2(rank_pos + 1)
    return dcg_sum

# -------------------------------------------------------
# 11. Compute NDCG for Single Query
# -------------------------------------------------------
def compute_query_ndcg(found_ids, target_ids, limit_k=5):
    actual_gains = []

    for doc_id in found_ids[:limit_k]:
        if doc_id in target_ids:
            actual_gains.append(1)
        else:
            actual_gains.append(0)

    ideal_gains = sorted(actual_gains, reverse=True)

    dcg_metric = compute_dcg(actual_gains)
    idcg_metric = compute_dcg(ideal_gains)

    if idcg_metric == 0:
        return 0.0

    return dcg_metric / idcg_metric

# -------------------------------------------------------
# 12. Compute Mean NDCG for Dataset
# -------------------------------------------------------
def evaluate_mean_ndcg(dataset, limit_k=5):
    ndcg_list = []

    for entry in dataset:
        q_text = entry["query"]
        t_ids = entry["relevant_ids"]

        top_hits = get_top_matches(q_text, limit_k=limit_k)
        hit_ids = [hit["id"] for hit in top_hits]

        ndcg_val = compute_query_ndcg(hit_ids, t_ids, limit_k=limit_k)
        ndcg_list.append(ndcg_val)

    if not ndcg_list:
        return 0.0

    return sum(ndcg_list) / len(ndcg_list)

# -------------------------------------------------------
# 13. Generate Chatbot Response Structure
# -------------------------------------------------------
def generate_bot_reply(user_query):
    results_list = get_top_matches(user_query, limit_k=5)

    if not results_list or results_list[0]["score"] <= 0:
        return "Sorry, I could not find a matching answer in the company knowledge base."

    primary_match = results_list[0]

    reply_msg = (
        f"Most relevant answer:\n"
        f"{primary_match['answer']}\n\n"
        f"Matched Question: {primary_match['question']}\n"
        f"BM25 Score: {primary_match['score']:.4f}"
    )

    # Cross-reference with Ground Truth
    target_ids = fetch_target_ids(user_query)

    if target_ids is not None:
        extracted_ids = [node["id"] for node in results_list]

        rr_score = compute_rr(extracted_ids, target_ids)
        ndcg_score = compute_query_ndcg(extracted_ids, target_ids, limit_k=5)

        reply_msg += (
            f"\nReciprocal Rank (RR): {rr_score:.4f}"
            f"\nNDCG: {ndcg_score:.4f}"
        )
    else:
        reply_msg += (
            "\nReciprocal Rank (RR): Not available"
            "\nNDCG: Not available"
            "\nReason: This query is not defined in the ground-truth test set."
        )

    return reply_msg

# -------------------------------------------------------
# 14. Present Dummy Query Outputs
# -------------------------------------------------------
def display_sample_outputs():
    print("\n================ SAMPLE PROMPT RESULTS ================\n")

    for test_prompt in eval_prompts:
        print(f"User Prompt: {test_prompt}")
        matches = get_top_matches(test_prompt, limit_k=3)

        for ranking, element in enumerate(matches, start=1):
            print(f"Rank {ranking}")
            print(f"Question : {element['question']}")
            print(f"Answer   : {element['answer']}")
            print(f"Score    : {element['score']:.4f}")
            print("-" * 50)

        print("=" * 60)

# -------------------------------------------------------
# 15. Run System Evaluation Reports
# -------------------------------------------------------
def run_system_evaluation():
    print("\n================ SEARCH EVALUATION ====================\n")

    mrr_metric = evaluate_mrr(test_ground_truth, limit_k=5)
    ndcg_metric = evaluate_mean_ndcg(test_ground_truth, limit_k=5)

    print(f"Overall MRR  (Mean Reciprocal Rank)       : {mrr_metric:.4f}")
    print(f"Overall NDCG (Average Ranking Quality)    : {ndcg_metric:.4f}")

    print("\nDetailed Per Query Evaluation:\n")

    for node in test_ground_truth:
        q_string = node["query"]
        t_ids = node["relevant_ids"]

        retrieved_nodes = get_top_matches(q_string, limit_k=5)
        retrieved_ids = [res["id"] for res in retrieved_nodes]

        rr_val = compute_rr(retrieved_ids, t_ids)
        ndcg_val = compute_query_ndcg(retrieved_ids, t_ids, limit_k=5)

        print(f"Query          : {q_string}")
        print(f"Retrieved IDs  : {retrieved_ids}")
        print(f"Relevant IDs   : {t_ids}")
        print(f"RR             : {rr_val:.4f}")
        print(f"NDCG           : {ndcg_val:.4f}")
        print("-" * 60)

def start_chat_session():
    print("\n================ COMPANY BM25 CHATBOT =================")
    print("Type 'exit' to stop the chatbot.\n")
    print("Try queries such as:")
    print("- leave policy")
    print("- work from home request")
    print("- office hours")
    print("- forgot email password")
    print("- travel reimbursement")
    print("- hr contact")
    print("- laptop issue")
    print("- company holidays\n")

    while True:
        user_msg = input("You: ").strip()

        if user_msg.lower() == "exit":
            print("Bot: Goodbye!")
            break

        bot_reply = generate_bot_reply(user_msg)
        print("\nBot:")
        print(bot_reply)
        print()

if __name__ == "__main__":
    display_sample_outputs()
    run_system_evaluation()
    start_chat_session()