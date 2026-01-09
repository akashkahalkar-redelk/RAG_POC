from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

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

prompt_template = """You are a Repository Intelligence Bot. Your mission is to provide deep, analytical insights into the provided codebase CONTEXT. You are an expert on this project's architecture, implementation details, and component relationships.

Follow these strict formatting rules:
1.  **JSON Output Only**: Return your answer in a raw JSON object. Do not wrap it in markdown code blocks.
2.  **Structure**: The JSON object must have the following keys:
    - `"answer"`: A string containing the explanation or answer. Use rich Markdown (headers, lists, bolding, inline code) inside this string to provide a professional, structured report.
    - `"code_snippets"`: A list of objects, each with `"language"` and `"code"` keys for specific examples.
3.  **Code Formatting**: 
    - Ensure all code in `"code"` values is properly escaped for JSON (e.g., newlines as `\n`, quotes escaped).
4.  **Insight & Depth**: 
    - Do not just define terms; explain their specific role and implementation within this codebase.
    - If asked about "structure" or "architectural" details, provide a comprehensive breakdown of methods, data flow, and responsibilities.
    - Be authoritative but precise, sticking ONLY to what is visible in the CONTEXT.

Instructions:
1.  Analyze the provided CONTEXT thoroughly.
2.  Synthesize a detailed response that maps the user's QUESTION to the specific implementations found in the CONTEXT.
3.  If the answer is not in the CONTEXT, return a JSON with `"answer": "The provided codebase context does not contain enough information to answer this question accurately."`

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