# LegalMind RAG 🏛️

A domain-specific **Retrieval-Augmented Generation (RAG)** system for legal document question answering. LegalMind retrieves relevant clauses from legal documents before generating responses — keeping answers grounded in source material and reducing hallucinations.

---

## 📌 Overview

Legal text is dense, citation-heavy, and unforgiving of inaccuracies. General-purpose LLMs often produce confident but incorrect answers when queried on legal content. LegalMind addresses this by combining semantic retrieval with a generative model — answers are always anchored to actual document passages.

**Domain:** ADA / Accessibility Law & Legal Contracts  
**Approach:** RAG pipeline with RAGAS-based evaluation  
**Interface:** Interactive Streamlit web app

---

## 🏗️ Architecture

```
User Query
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
    │
    ▼
ChromaDB Vector Store  ◄──── Legal Documents (data/)
    │
    ▼
Top-k Relevant Chunks (chunk_overlap=50)
    │
    ▼
LLM: flan-t5-base
    │
    ▼
Grounded Answer
```

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Language Model | `flan-t5-base` |
| Frontend | Streamlit |
| Evaluation | RAGAS Framework |

---

## 📊 Evaluation Results (RAGAS)

Evaluated on legal domain Q&A (ADA / accessibility law documents):

| Metric | Score |
|---|---|
| Faithfulness | 0.57 |
| Answer Relevancy | 0.53 |

> Faithfulness measures how grounded the answer is in the retrieved context. Answer Relevancy measures how well the answer addresses the question.

---

## 📁 Project Structure

```
legalmind-rag/
├── data/               # Legal documents (PDFs / text files)
├── src/                # Core RAG pipeline modules
├── vectorstore/        # Persisted ChromaDB vector index
├── app.py              # Streamlit application entry point
└── requirements.txt    # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/karthik0987/legalmind-rag.git
cd legalmind-rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 💡 How It Works

1. **Ingestion** — Legal documents are loaded, split into overlapping chunks (`chunk_overlap=50`), and embedded using HuggingFace sentence transformers.
2. **Indexing** — Embeddings are stored in a persistent ChromaDB vector store.
3. **Retrieval** — On each query, the top-k most semantically similar chunks are retrieved.
4. **Generation** — Retrieved chunks are passed as context to `flan-t5-base`, which generates a grounded response.
5. **Evaluation** — Pipeline quality is measured using the RAGAS framework (faithfulness + answer relevancy).

---

## 🔮 Future Improvements

- [ ] Upgrade LLM to Groq-hosted LLaMA 3 for faster inference
- [ ] Add context window reranking (Cohere / cross-encoder)
- [ ] Expand evaluation to include context recall and precision
- [ ] Support multi-document citation with source highlighting
- [ ] Deploy on Streamlit Cloud / HuggingFace Spaces

---

## 👤 Author

**Karthik** — MS Computer Science, University of North Texas  
GitHub: [@karthik0987](https://github.com/karthik0987)  
LinkedIn: [linkedin.com/in/karthik](https://linkedin.com/in/karthik)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
