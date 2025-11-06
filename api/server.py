from fastapi import FastAPI
from pydantic import BaseModel
from retrieval.query_engine import get_answer

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QueryRequest):
    result = get_answer(request.question)
    answer_text = result["result"]
    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            sources.append({
                "file": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:300] + "..."  # show preview
            })

    return {
        "answer_markdown": f"### Answer\n\n{answer_text}",
        "sources": sources
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def init():
    return {"status": "startup"}