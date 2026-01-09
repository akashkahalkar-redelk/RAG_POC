from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from retrieval.query_engine import get_answer

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QueryRequest):
    result = get_answer(request.question)
    answer_data = result["result"] # Now a dictionary
    
    # Ensure answer_data is a dict (fallback safety)
    if isinstance(answer_data, str):
         answer_data = {"answer": answer_data, "code_snippets": []}

    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            sources.append({
                "file": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:300] + "..."  # show preview
            })

    return {
        "answer": answer_data.get("answer", ""),
        "code_snippets": answer_data.get("code_snippets", []),
        "sources": sources
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def init():
    return {"status": "startup"}