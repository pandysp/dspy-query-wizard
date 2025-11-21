from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Welcome to the DSPy Query Wizard API"}

@app.post("/api/query")
def query(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    # Placeholder for logic
    return {
        "question": request.question,
        "human_answer": "Not implemented yet",
        "machine_answer": "Not implemented yet"
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)