import streamlit as st
from src.generator import generate_answer
from src.embeddings import build_vectorstore
from src.ingest import load_and_chunk
import os

st.set_page_config(page_title="LegalMind", layout="wide")
st.title("LegalMind - Legal Document Q&A")
st.markdown("Ask any question about the loaded legal document.")

if not os.path.exists("vectorstore"):
    with st.spinner("Building knowledge base..."):
        chunks = load_and_chunk("data/legal.pdf")
        build_vectorstore(chunks)
    st.success("Knowledge base ready!")

query = st.text_input(
    "Ask a legal question:", placeholder="e.g. What are the ADA requirements?"
)

if query:
    with st.spinner("Thinking..."):
        answer, docs = generate_answer(query)
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("View Retrieved Context Chunks"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}** (Page {doc.metadata.get('page', 'N/A')})")
            st.write(doc.page_content)
            st.divider()
