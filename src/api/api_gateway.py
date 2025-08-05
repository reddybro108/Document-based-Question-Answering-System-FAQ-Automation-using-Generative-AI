
from fastapi import FastAPI
from pydantic import BaseModel
from src.core.qa_engine import DocumentQAEngine

app = FastAPI()
qa_engine = DocumentQAEngine()

class QuestionRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    answer = qa_engine.ask(req.query)
    return {"question": req.query, "answer": answer}