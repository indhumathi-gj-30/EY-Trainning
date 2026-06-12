import os
from anthropic import Anthropic
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

delivery_faqs = [
    "What are the delivery hours? We operate daily from 7:00 AM to midnight.",
    "Is there a minimum order value? No, but orders below $10 incur a small cart fee.",
    "How can I track my rider? View the live map in the 'Active Orders' section.",
    "Can I change my delivery address? Only before the restaurant accepts your order.",
    "What if my food is cold? Contact live chat within 15 minutes for a refund or redelivery.",
    "Do you offer contact-free delivery? Yes, select 'Leave at my door' during checkout.",
    "How do I apply a promo code? Enter the voucher code on the payment summary screen.",
    "Why was my order cancelled? This happens if the kitchen closes or no riders are nearby.",
    "Can I schedule an order? Yes, pre-order up to 3 days in advance via the app.",
    "What payment options exist? We support wallets, net banking, and cash on delivery."
]

encoder_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
kb_vector_idx = FAISS.from_texts(delivery_faqs, encoder_model)
anthropic_mgr = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("\n🍔 Foodie Support System Online!")
print("Type your question below (or 'exit' to quit):\n")

while True:
    client_query = input("User: ")

    if not client_query.strip() or client_query.strip().lower() == "exit":
        print("Stopping assistant...")
        break

    
    extracted_chunks = kb_vector_idx.similarity_search_with_score(client_query, k=2)
    context_str = "\n".join([chunk.page_content for chunk, _ in extracted_chunks])
    prompt_payload = f"""
Answer ONLY using the FAQ context below.

FAQ Context:
{context_str}

Question:
{client_query}

Provide a concise answer.
"""
    completion_data = anthropic_mgr.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt_payload}]
    )

    print(f"\nBot: {completion_data.content[0].text}\n")
    print("[Evaluation Metrics: Proximity Delta]")
    for chunk, proximity_weight in extracted_chunks:
        print(f" -> Delta: {proximity_weight:.4f} | Source: {chunk.page_content}")
    print("=" * 60 + "\n")