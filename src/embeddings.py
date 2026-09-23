from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# BGE outperforms MiniLM on retrieval benchmarks (MTEB) and is still small
# enough to run on CPU. It needs an instruction prefix on queries (not on
# documents) to get its full retrieval quality.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEQueryEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str):
        return super().embed_query(QUERY_INSTRUCTION + text)


def get_embeddings():
    return BGEQueryEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks):
    print("Loading embedding model...")
    embeddings = get_embeddings()
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("Vector store saved to /vectorstore")
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return vectorstore
