from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx
import sys
import os

# Ensure backend package is in path if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the refactored retriever logic
try:
    from backend.retriever import fetch_colbert_results
except ImportError:
    # Fallback for direct execution
    from retriever import fetch_colbert_results

app = FastAPI()

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
         raise HTTPException(status_code=503, detail=f"Retriever service failed: {str(e)}")

    return {
        "question": request.question,
        "retrieved_contexts": retrieved_text,
        "human_answer": "Not implemented yet",
        "machine_answer": "Not implemented yet"
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
