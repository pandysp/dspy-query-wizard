from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx
from contextlib import asynccontextmanager

# Import the refactored retriever logic
from backend.retriever import fetch_colbert_results, prewarm_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await prewarm_cache()
    yield
    # Shutdown (optional cleanup)


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
async def home():
    return {"message": "Welcome to the DSPy Query Wizard API"}


@app.post("/api/query")
async def query(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")

    try:
        async with httpx.AsyncClient() as client:
            # Use the robust fetch function (handles ColBERT -> Wikipedia fallback)
            raw_results = await fetch_colbert_results(client, request.question, k=3)

        retrieved_text = [r.get("text", str(r)) for r in raw_results]

    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Retriever service failed: {str(e)}"
        )

    return {
        "question": request.question,
        "retrieved_contexts": retrieved_text,
        "human_answer": "Not implemented yet",
        "machine_answer": "Not implemented yet",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
