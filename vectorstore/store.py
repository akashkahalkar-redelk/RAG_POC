# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from embedding.embedder import get_embedder

def build_store(documents, persist_directory="db"):
    embeddings = get_embedder()
    
    BATCH_SIZE = 100
    total_docs = len(documents)
    print(f"Starting ingestion of {total_docs} chunks in batches of {BATCH_SIZE}...", flush=True)
    
    vectordb = None
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(total_docs + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} chunks)...", flush=True)
        
        if vectordb is None:
            vectordb = Chroma.from_documents(batch, embeddings, persist_directory=persist_directory)
        else:
            vectordb.add_documents(batch)
            
    print("Persisting database...")
    # if vectordb:
    #     vectordb.persist()  # Deprecated in newer Chroma versions, auto-persists
    print("Ingestion complete.")
    return vectordb

def load_store(persist_directory="db"):
    embeddings = get_embedder()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)