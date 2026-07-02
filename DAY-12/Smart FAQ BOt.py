from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os


faq_data = [
    "What is a REST API? A REST API enables communication between applications using HTTP methods.",
    "What is FastAPI? FastAPI is a modern Python framework for building APIs.",
    "What is Docker? Docker packages applications and dependencies into containers.",
    "What is CI/CD? CI/CD automates software testing and deployment processes.",
    "What is PostgreSQL? PostgreSQL is an open-source relational database.",
    "What is Git? Git is a distributed version control system.",
    "What is a microservice? A microservice is a small independent service within an application.",
    "What is JWT authentication? JWT provides secure token-based authentication.",
    "What is caching? Caching stores frequently accessed data for faster retrieval.",
    "What is Kubernetes? Kubernetes automates container deployment and management."
]

embedding_engine = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_texts(
    faq_data,
    embedding_engine
)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def retrieve_faqs(question):

    search_results = vector_db.similarity_search_with_score(
        question,
        k=2
    )

    faq_context = "\n".join(
        doc.page_content for doc, _ in search_results
    )

    return faq_context, search_results

def generate_response(question, faq_context):

    prompt = f"""
You are a Smart FAQ Bot.

Answer the question only using the FAQ information below.

FAQ Information:
{faq_context}

Question:
{question}

Provide a concise and accurate answer.
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=120
    )

    return completion.choices[0].message.content
def start_chat():

    print("\n===================================")
    print("        SMART FAQ BOT")
    print("===================================")
    print("Type 'exit' to quit\n")

    while True:

        user_question = input("You: ")

        if user_question.lower() == "exit":
            print("\nSession Closed.")
            break

        faq_context, matches = retrieve_faqs(
            user_question
        )

        answer = generate_response(
            user_question,
            faq_context
        )

        print("\nBot:", answer)

        print("\nTop Retrieved FAQs:")

        for doc, score in matches:
            print(
                f"Similarity Score: {score:.4f}"
            )
            print(
                f"FAQ: {doc.page_content}"
            )

        print("-" * 60)

if __name__ == "__main__":
    start_chat()