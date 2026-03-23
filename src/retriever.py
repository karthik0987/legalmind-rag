from src.embeddings import load_vectorstore

def get_retriever(k=4):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

def retrieve(query: str, k=4):
    retriever = get_retriever(k)
    docs = retriever.invoke(query)
    return docs
