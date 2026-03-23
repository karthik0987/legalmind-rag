from src.generator import generate_answer
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def cosine_score(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return round(float(util.cos_sim(emb1, emb2)), 4)


def run_evaluation():
    test_questions = [
        "What are the ADA requirements for state governments?",
        "What is the purpose of the ADA transition plan?",
        "What are the self-evaluation requirements under ADA?",
        "How should grievance procedures be handled under ADA?",
        "What are the physical accessibility requirements?",
    ]

    print("\n===== LegalMind RAG Evaluation =====\n")
    faithfulness_scores = []
    relevancy_scores = []

    for q in test_questions:
        answer, docs = generate_answer(q)
        context = " ".join([doc.page_content for doc in docs])
        faithfulness = cosine_score(answer, context)
        relevancy = cosine_score(answer, q)
        faithfulness_scores.append(faithfulness)
        relevancy_scores.append(relevancy)
        print(f"Q: {q}")
        print(f"A: {answer[:120]}...")
        print(f"Faithfulness: {faithfulness} | Relevancy: {relevancy}\n")

    avg_faith = round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
    avg_relev = round(sum(relevancy_scores) / len(relevancy_scores), 4)

    print("===== FINAL SCORES =====")
    print(f"Avg Faithfulness:     {avg_faith}")
    print(f"Avg Answer Relevancy: {avg_relev}")


if __name__ == "__main__":
    run_evaluation()
