from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from chunking.splitter import chunk_documents
from ingestion.loader import load_codebase
from vectorstore.store import build_store
from vectorstore.store import load_store
import hashlib
import os

def get_persist_dir():
    path = os.environ.get("PROJECT_PATH")
    if not path:
        return "db"
    if path.startswith("~"):
        path = os.path.expanduser(path)
    
    # Create a unique hash for the project path
    path_hash = hashlib.md5(path.encode()).hexdigest()
    return f"db/chroma_{path_hash}"

PERSIST_DIR = get_persist_dir()
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

Follow these strict formatting rules:
1.  **JSON Output Only**: Return your answer in a raw JSON object. Do not wrap it in markdown code blocks.
2.  **Structure**: The JSON object must have the following keys:
    - `"answer"`: A string containing the explanation or answer.
    - `"code_snippets"`: A list of objects, each with `"language"` and `"code"` keys.
3.  **Code Formatting**: 
    - Ensure all code in `"code"` values is properly escaped for JSON (e.g., newlines as `\n`, quotes indented).
    - Do not use markdown format in the `"answer"` string.
4.  **Conciseness**: Be direct.

Instructions:
1.  Analyze the CONTEXT snippets.
2.  Synthesize an answer exclusively from the CONTEXT.
3.  If the answer is not in the CONTEXT, return a JSON with `"answer": "I cannot find the answer in the provided context."`

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
llm =  OllamaLLM(model="qwen2.5-coder:7b", temperature=0.1)

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

import json
import re

# ---- Step 2: Answer query ----
def get_answer(question: str) -> dict:
    res = qa({"query": question})
    print("\n--- Retrieved Context ---")
    for doc in res.get("source_documents", []):
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:500]}...")
    print("--- End of Context ---\n")
    
    # Parse the LLM output into a dictionary
    raw_answer = res.get("result", "")
    try:
        # Strip markdown code blocks if present
        cleaned_answer = re.sub(r'^```json\s*', '', raw_answer.strip(), flags=re.MULTILINE)
        cleaned_answer = re.sub(r'^```\s*', '', cleaned_answer.strip(), flags=re.MULTILINE)
        cleaned_answer = re.sub(r'\s*```$', '', cleaned_answer.strip(), flags=re.MULTILINE)
        
        parsed_answer = json.loads(cleaned_answer)
        res["result"] = parsed_answer
    except json.JSONDecodeError:
        print(f"Failed to parse JSON answer. Raw: {raw_answer}")
        res["result"] = {
            "answer": raw_answer,
            "code_snippets": []
        }
        
    return res