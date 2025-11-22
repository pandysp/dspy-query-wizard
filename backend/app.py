from fastapi.applications import FastAPI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import os
import logging
import json
import asyncio
import dspy  # type: ignore
from dspy.streaming import StreamListener, StatusMessageProvider, StreamResponse, StatusMessage # type: ignore
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Import the refactored retriever logic and RAG modules
from backend.retriever import prewarm_cache, search_wikipedia
from backend.rag import HumanRAG, MachineRAG, AgenticRAG, AgenticSignature

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global instances
human_rag: HumanRAG | None = None
machine_rag: MachineRAG | None = None
agentic_rag: AgenticRAG | None = None
openai_client: AsyncOpenAI | None = None


def configure_lm() -> None:
    """Configures the Language Model."""
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. Machine RAG will fail.")
        return

    openai_client = AsyncOpenAI(api_key=api_key)

    model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    if not model_name.startswith("openai/"):
        full_model_name = f"openai/{model_name}"
    else:
        full_model_name = model_name

    # Use Chat API for better streaming support
    lm = dspy.LM(
        full_model_name, 
        api_key=api_key,
        model_type="chat",
    )
    dspy.settings.configure(lm=lm)
    logger.info(f"LM configured: {full_model_name} (Chat API)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global human_rag, machine_rag, agentic_rag

    # Startup
    configure_lm()

    logger.info("Initializing RAG pipelines...")
    human_rag = HumanRAG()
    machine_rag = MachineRAG()
    agentic_rag = AgenticRAG()

    # Try to load compiled MachineRAG
    compiled_path = os.path.join(
        os.path.dirname(__file__), "data", "compiled_machine_rag.json"
    )
    if os.path.exists(compiled_path):
        try:
            logger.info(f"Loading compiled MachineRAG from {compiled_path}...")
            machine_rag.load(compiled_path)
            logger.info("MachineRAG loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load compiled MachineRAG: {e}")
    else:
        logger.warning("No compiled MachineRAG found. Using unoptimized version.")
        
    # Try to load compiled AgenticRAG
    compiled_agentic_path = os.path.join(
        os.path.dirname(__file__), "data", "compiled_agentic_rag.json"
    )
    if os.path.exists(compiled_agentic_path):
        try:
            logger.info(f"Loading compiled AgenticRAG from {compiled_agentic_path}...")
            agentic_rag.load(compiled_agentic_path)
            logger.info("AgenticRAG loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load compiled AgenticRAG: {e}")


    logger.info("Pre-warming cache...")
    await prewarm_cache()

    yield
    # Shutdown (optional cleanup)


app: FastAPI = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    manual_queries: list[str] | None = None


@app.get("/")
async def home():
    return {"message": "Welcome to the DSPy Query Wizard API"}


@app.post("/api/query")
async def query(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")

    if human_rag is None or machine_rag is None or agentic_rag is None:
        raise HTTPException(status_code=503, detail="RAG pipelines not initialized")

    try:
        # Run pipelines
        # TODO: Run in parallel for performance

        # Human RAG (simulated human effort if manual_queries provided)
        human_pred = human_rag(request.question, queries=request.manual_queries)

        # Machine RAG
        machine_pred = machine_rag(request.question)

        # Agentic RAG (The "Smart" approach)
        def run_agentic_rag_sync(q: str):
            return agentic_rag(q)

        agentic_pred = await asyncio.to_thread(run_agentic_rag_sync, request.question)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Pipeline execution failed: {str(e)}"
        )

    return {
        "question": request.question,
        "human_answer": {"answer": human_pred.answer, "context": human_pred.context},
        "machine_answer": {
            "answer": machine_pred.answer,
            "context": machine_pred.context,
            "search_query": getattr(machine_pred, "search_query", None),
        },
        "agentic_answer": {
            "answer": agentic_pred.answer,
            "context": getattr(
                agentic_pred, "history", []
            ),  # ReAct history contains steps
        },
    }

# --- Streaming Chat API ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequestPayload(BaseModel):
    messages: list[ChatMessage]
    system_prompt: str | None = None
    # Optional: explicit mode selector if system_prompt isn't enough
    mode: str | None = None 


class VercelStatusMessageProvider(StatusMessageProvider):
    """
    Maps DSPy status updates to Vercel AI SDK streaming protocol.
    We return 'raw' JSON objects that stream_dspy_generator will then format into SSE.
    """
    def tool_start_status_message(self, instance, inputs):
        # Return raw data; stream_dspy_generator will add toolCallId and format to SSE.
        return json.dumps({
            "type": "tool-input-start-raw",
            "toolName": instance.name,
            "inputs": inputs,
        })

    def tool_end_status_message(self, outputs):
        return json.dumps({
            "type": "tool-output-available-raw",
            "outputs": outputs,
        })


async def stream_dspy_generator(stream_gen):
    """Helper to iterate DSPy async generator and yield Vercel formatted SSE chunks."""
    import uuid
    import traceback
    
    message_id = f"msg_{uuid.uuid4().hex}"
    text_id = f"text_{uuid.uuid4().hex}" # For the main answer text block
    
    reasoning_id = None
    current_tool_call_id = None
    
    # Send initial message start part
    yield f'data: {json.dumps({"type": "start", "messageId": message_id})}\n'
    
    try:
        # Send text-start for the main text block
        yield f'data: {json.dumps({"type": "text-start", "id": text_id})}\n'

        async for chunk in stream_gen:
            
            if isinstance(chunk, StreamResponse):
                # Reasoning Part (next_thought, rationale, reasoning)
                if chunk.signature_field_name in ["reasoning", "next_thought", "rationale"]:
                    # Close current text block if active
                    if reasoning_id is None:
                        reasoning_id = f"reasoning_{uuid.uuid4().hex[:8]}"
                        yield f'data: {json.dumps({"type": "reasoning-start", "id": reasoning_id})}\n'
                    
                    if chunk.chunk:
                        yield f'data: {json.dumps({"type": "reasoning-delta", "id": reasoning_id, "delta": chunk.chunk})}\n'
                
                # Standard Text output (answer field)
                else:
                    # Close reasoning if active before emitting text
                    if reasoning_id:
                        yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_id})}\n'
                        reasoning_id = None
                    
                    if chunk.chunk:
                         yield f'data: {json.dumps({"type": "text-delta", "id": text_id, "delta": chunk.chunk})}\n'
            
            elif isinstance(chunk, StatusMessage):
                # Close reasoning if active before emitting status
                if reasoning_id:
                    yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_id})}\n'
                    reasoning_id = None
                
                # Parse raw status data and map to Vercel Tool events
                try:
                    status_data = json.loads(chunk.message)
                    status_type = status_data.get("type")
                    
                    if status_type == "tool-input-start-raw":
                        current_tool_call_id = f"call_{uuid.uuid4().hex}"
                        yield f'data: {json.dumps({
                            "type": "tool-input-start",
                            "toolCallId": current_tool_call_id,
                            "toolName": status_data["toolName"]
                        })}\n'
                        yield f'data: {json.dumps({
                            "type": "tool-input-available",
                            "toolCallId": current_tool_call_id,
                            "toolName": status_data["toolName"],
                            "input": status_data["inputs"] # Pass original inputs
                        })}\n'
                        
                    elif status_type == "tool-output-available-raw":
                        if current_tool_call_id:
                            yield f'data: {json.dumps({
                                "type": "tool-output-available",
                                "toolCallId": current_tool_call_id,
                                "output": status_data["outputs"]
                            })}\n'
                        current_tool_call_id = None # Reset for next tool call
                        
                    else: # Fallback for other custom status messages as generic data
                        yield f'data: {json.dumps({"type": "data", "data": {"type": "status", "message": chunk.message}})}\n'
                        
                except json.JSONDecodeError:
                    yield f'data: {json.dumps({"type": "data", "data": {"type": "status", "message": chunk.message}})}\n'
            
            # Fallback: Handle final Prediction object if streaming didn't capture tokens
            elif isinstance(chunk, dspy.primitives.prediction.Prediction):
                logger.info("Received final Prediction object (fallback).")
                # Close reasoning if active
                if reasoning_id:
                    yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_id})}\n'
                    reasoning_id = None
                
                # Emit delta for the answer from fallback, then end the text block
                if chunk.answer:
                    yield f'data: {json.dumps({"type": "text-delta", "id": text_id, "delta": chunk.answer})}\n'
            
        # Final cleanup and termination messages
        if reasoning_id:
            yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_id})}\n'
            
        # Ensure the main text block is ended
        yield f'data: {json.dumps({"type": "text-end", "id": text_id})}\n'
        
        # Finish message
        yield f'data: {json.dumps({"type": "finish"})}\n'
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        traceback.print_exc()
        yield f'data: {json.dumps({"type": "error", "errorText": f"Streaming error: {str(e)}"})}\n'
        
    finally:
        # Stream termination marker
        yield 'data: [DONE]\n'


async def stream_human_mode(messages: list[ChatMessage], system_prompt: str):
    """
    Human Mode: Uses dspy.ReAct but with the User's System Prompt as the instruction.
    """
    question = messages[-1].content
    
    class DynamicSignature(dspy.Signature):
        __doc__ = system_prompt 
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # 2. Create ReAct Module
    # We use the same tools (search_wikipedia)
    react = dspy.ReAct(DynamicSignature, tools=[search_wikipedia])
    
    # 3. Setup Streaming
    stream_listeners = [
        dspy.streaming.StreamListener(signature_field_name="answer"),
        # Listen to thoughts/reasoning (ReAct usually uses 'next_thought')
        dspy.streaming.StreamListener(signature_field_name="next_thought", allow_reuse=True),
        dspy.streaming.StreamListener(signature_field_name="reasoning", allow_reuse=True),
    ]
    
    stream_react = dspy.streamify(
        react,
        stream_listeners=stream_listeners,
        status_message_provider=VercelStatusMessageProvider(),
    )
    
    output_stream = stream_react(question=question)
    
    async for chunk in stream_dspy_generator(output_stream):
        yield chunk


async def stream_machine_mode(messages: list[ChatMessage]):
    """
    Machine Mode: Uses the pre-compiled AgenticRAG (dspy.ReAct).
    """
    if not agentic_rag:
        yield '0:Error: AgenticRAG not initialized.\n'
        return

    question = messages[-1].content
    react_module = agentic_rag.react
    
    stream_listeners = [
        dspy.streaming.StreamListener(signature_field_name="answer"),
        dspy.streaming.StreamListener(signature_field_name="next_thought", allow_reuse=True),
        dspy.streaming.StreamListener(signature_field_name="reasoning", allow_reuse=True),
    ]
    
    stream_react = dspy.streamify(
        react_module,
        stream_listeners=stream_listeners,
        status_message_provider=VercelStatusMessageProvider(),
    )

    output_stream = stream_react(question=question)
    
    prompt_data = {
        "type": "dspy-prompt",
        "messages": [{"role": "system", "content": "Optimized Agentic ReAct Pipeline"}], 
        "info": "Running compiled dspy.ReAct module with automated tool use."
    }
    yield f'2:[{json.dumps(prompt_data)}]\n'

    async for chunk in stream_dspy_generator(output_stream):
        yield chunk


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequestPayload):
    response = StreamingResponse(
        stream_human_mode(request.messages, request.system_prompt) 
        if request.system_prompt 
        else stream_machine_mode(request.messages),
        media_type="text/event-stream" 
    )
    # Helper header for Data Stream Protocol
    response.headers['x-vercel-ai-ui-message-stream'] = 'v1'
    # Important for SSE: disable chunking/buffering from reverse proxies (like Nginx, Gunicorn)
    response.headers['Cache-Control'] = 'no-cache, no-transform'
    response.headers['Connection'] = 'keep-alive'
    return response