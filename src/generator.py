from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.retriever import retrieve

MODEL_NAME = "google/flan-t5-base"

print("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

NO_ANSWER_MSG = "The provided document does not contain enough information to answer this question."

PROMPT_TEMPLATE = """You are a legal assistant. Answer the question using ONLY the excerpts below.

Rules:
- Do not use any knowledge beyond the excerpts.
- If the excerpts don't contain enough information, respond exactly: "{no_answer}"
- Be precise and concise.
- Cite the page number(s) your answer relies on, e.g. (p. 3).

Excerpts:
{context}

Question: {query}

Answer:"""


def format_context(docs):
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}] {doc.page_content}" for doc in docs
    )


def generate_answer(query: str):
    docs = retrieve(query)
    context = format_context(docs)
    prompt = PROMPT_TEMPLATE.format(no_answer=NO_ANSWER_MSG, context=context, query=query)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, docs

if __name__ == "__main__":
    query = "What are the ADA requirements for state governments?"
    answer, docs = generate_answer(query)
    print(f"Answer: {answer}")
