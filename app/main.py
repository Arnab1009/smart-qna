from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from app.rag_chain import RAGChain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://arnab1009.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class QuestionRequest(BaseModel):
    question: str = Field(..., example="What are attention mechanisms in transformers?")
    k: Optional[int] = Field(None, example=5)
    llm_model: Optional[str] = Field(None, example="gemini-2.5-pro-exp-03-25")
    embedding_model: Optional[str] = Field(None, example="text-embedding-005")

@app.post("/ask")
def ask_question(input: QuestionRequest):
    try: 
    # Use provided values or defaults
        rag = RAGChain(
            k=input.k,
            model_name=input.llm_model,
            embedding_model_name=input.embedding_model
        )

        answer, sources = rag.run(input.question)

        return {
            "question": input.question,
            "answer": answer,
            "sources": sources,
            "top_k": rag.k
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}
