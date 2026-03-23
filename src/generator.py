from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.retriever import retrieve

print("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(query: str):
    docs = retrieve(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a legal assistant. Use the context below to answer the question accurately.

Context:
{context}

Question: {query}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, docs

if __name__ == "__main__":
    query = "What are the ADA requirements for state governments?"
    answer, docs = generate_answer(query)
    print(f"Answer: {answer}")
