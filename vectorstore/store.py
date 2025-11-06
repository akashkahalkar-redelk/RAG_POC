# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from embedding.embedder import get_embedder

def build_store(documents, persist_directory="db"):
    embeddings = get_embedder()
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def load_store(persist_directory="db"):
    embeddings = get_embedder()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

