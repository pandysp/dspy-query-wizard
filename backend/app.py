from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import os
import logging

# Import the refactored retriever logic and RAG modules
from backend.retriever import prewarm_cache, configure_retriever
from backend.rag import HumanRAG, MachineRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
human_rag: HumanRAG | None = None
machine_rag: MachineRAG | None = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    global human_rag, machine_rag
    
    # Startup
    logger.info("Configuring retriever...")
    configure_retriever()
    
    logger.info("Initializing RAG pipelines...")
    human_rag = HumanRAG()
    machine_rag = MachineRAG()
    
    # Try to load compiled MachineRAG
    compiled_path = os.path.join(os.path.dirname(__file__), "data", "compiled_machine_rag.json")
    if os.path.exists(compiled_path):
        try:
            logger.info(f"Loading compiled MachineRAG from {compiled_path}...")
            machine_rag.load(compiled_path)
            logger.info("MachineRAG loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load compiled MachineRAG: {e}")
    else:
        logger.warning("No compiled MachineRAG found. Using unoptimized version.")

    logger.info("Pre-warming cache...")
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

    if human_rag is None or machine_rag is None:
         raise HTTPException(status_code=503, detail="RAG pipelines not initialized")

    try:
        # Run pipelines
        # TODO: Run in parallel for performance using asyncio.to_thread if they are blocking/sync
        # DSPy modules are synchronous by default.
        
        # Human RAG
        human_pred = human_rag(request.question)
        
        # Machine RAG
        machine_pred = machine_rag(request.question)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Pipeline execution failed: {str(e)}"
        )

    return {
        "question": request.question,
        "human_answer": {
            "answer": human_pred.answer,
            "context": human_pred.context
        },
        "machine_answer": {
            "answer": machine_pred.answer,
            "context": machine_pred.context,
            "search_query": getattr(machine_pred, "search_query", None)
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)