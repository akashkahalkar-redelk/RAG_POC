from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from chunking.splitter import chunk_documents
from ingestion.loader import load_codebase
from vectorstore.store import build_store
from vectorstore.store import load_store
import os

PERSIST_DIR = "db"
is_ready = False

# ---- Step 1: Build & persist embeddings (only once) ----
if not os.path.exists(PERSIST_DIR):   # build once if not exists
    documents = load_codebase()
    chunks = chunk_documents(documents)
    vectordb = build_store(chunks, PERSIST_DIR)
else:
    vectordb = load_store(PERSIST_DIR)
    # retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    # docs = retriever.get_relevant_documents("Where is the database initialized?")
    # print([doc.page_content for doc in docs])
    is_ready = True

prompt_template = """You are an expert coding assistant. Your task is to answer the user's QUESTION based ONLY on the provided CONTEXT.

Follow these rules:
1.  Analyze the CONTEXT carefully. It contains snippets of code and documentation.
2.  Formulate your answer by extracting relevant information directly from the CONTEXT.
3.  If the CONTEXT does not contain the answer, you MUST state that you cannot find the answer in the provided information. Do not use any outside knowledge.
4.  Present code examples in Markdown code blocks.
5.  Your answer should be clear, concise, and directly address the QUESTION.

CONTEXT:
---
{context}
---

QUESTION:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# Using a lower temperature to make the model more factual and less creative
llm =  OllamaLLM(model="codellama:13b", temperature=0.1)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    # Using MMR (Maximal Marginal Relevance) to get more diverse results.
    # k: number of documents to return
    # fetch_k: number of documents to fetch initially to rerank
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20}),
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ---- Step 2: Answer query ----
def get_answer(question: str) -> dict:
    res = qa({"query": question})
    print("\n--- Retrieved Context ---")
    for doc in res.get("source_documents", []):
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:500]}...")
    print("--- End of Context ---\n")
    return res