import os
import json
import uuid
import dspy
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from .utils.prompt import ClientMessage
from .utils.tools import get_current_weather


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Request(BaseModel):
    messages: List[ClientMessage]


def configure_lm() -> None:
    """Configures the Language Model with streaming enabled."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found. DSPy will fail.")
        return

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not model_name.startswith("openai/"):
        full_model_name = f"openai/{model_name}"
    else:
        full_model_name = model_name

    # Configure LM - cache=False to ensure fresh responses during development
    lm = dspy.LM(full_model_name, api_key=api_key, cache=False)
    dspy.settings.configure(lm=lm)
    print(f"LM configured: {full_model_name}")


configure_lm()

# Initialize DSPy ReAct module with weather tool
react = dspy.ReAct("question->answer", tools=[get_current_weather])


class DSPyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    """Custom status message provider to track tool calls and LM operations."""

    def __init__(self):
        self.tool_calls = []
        self.current_tool_id = None

    def tool_start_status_message(self, instance, inputs):
        """Called when a tool starts executing."""
        tool_id = str(uuid.uuid4())
        self.current_tool_id = tool_id

        # Extract the actual arguments from inputs dict
        # DSPy passes inputs as a dict with parameter names as keys
        args = inputs if isinstance(inputs, dict) else {"input": inputs}

        self.tool_calls.append(
            {
                "id": tool_id,
                "name": instance.name,
                "args": args,
                "started": True,
                "completed": False,
            }
        )
        return f"tool_start:{tool_id}:{instance.name}:{json.dumps(args)}"

    def tool_end_status_message(self, outputs):
        """Called when a tool finishes executing."""
        if self.current_tool_id:
            for tool_call in self.tool_calls:
                if (
                    tool_call["id"] == self.current_tool_id
                    and not tool_call["completed"]
                ):
                    tool_call["result"] = outputs
                    tool_call["completed"] = True
                    return f"tool_end:{self.current_tool_id}:{json.dumps(outputs)}"
        return None

    def lm_start_status_message(self, instance, inputs):
        """Called when an LM call starts."""
        return "lm_start"

    def lm_end_status_message(self, outputs):
        """Called when an LM call ends."""
        return "lm_end"


async def stream_dspy_text(messages: List[ClientMessage], protocol: str = "data"):
    """
    Stream DSPy ReAct responses conforming to Vercel AI protocol.

    Streams:
    - Reasoning/thought tokens as they're generated
    - Tool call invocations and results
    - Execution status messages
    - Final answer

    Protocol types:
    - "text": Plain text chunks (all content concatenated)
    - "data": Structured stream with tool calls (Vercel AI SDK format)
    """
    print(f"[DEBUG] stream_dspy_text called with protocol={protocol}")

    # Extract the last user message as the question
    question = None
    for msg in reversed(messages):
        if msg.role == "user":
            for part in msg.parts:
                if part.type == "text" and part.text:
                    question = part.text
                    break
            if question:
                break

    print(f"[DEBUG] Extracted question: {question}")

    if not question:
        if protocol == "text":
            yield "No question found in messages."
        else:
            # SSE format error
            text_id = f"text_{uuid.uuid4().hex[:16]}"
            yield f'data: {{"type":"text-start","id":"{text_id}"}}\n\n'
            yield f'data: {{"type":"text-delta","id":"{text_id}","delta":"No question found in messages."}}\n\n'
            yield f'data: {{"type":"text-end","id":"{text_id}"}}\n\n'
            yield f'data: {{"type":"finish"}}\n\n'
            yield "data: [DONE]\n\n"
        return

    try:
        # Create status message provider to track tool calls and LM operations
        status_provider = DSPyStatusMessageProvider()

        # Listen to the "answer" field - ReAct's primary output field
        # Note: ReAct's internal Thought/Action/Observation cycles happen during
        # LM calls, and we'll track those via tool call status messages.
        # The answer field streams the final response tokens.
        stream_listeners = [
            dspy.streaming.StreamListener(signature_field_name="answer"),
        ]

        print("[DEBUG] Creating streamified ReAct module...")

        # Wrap the ReAct module with streaming
        stream_react = dspy.streamify(
            react,
            stream_listeners=stream_listeners,
            status_message_provider=status_provider,
            async_streaming=True,
        )

        print("[DEBUG] Executing streaming program...")

        # Execute the streaming program
        output_stream = stream_react(question=question)

        print("[DEBUG] Starting to iterate over output stream...")
        chunk_count = 0

        # Track tool calls for the data protocol
        tool_calls_pending = {}  # tool_id -> {name, args}
        tool_calls_completed = set()  # Track which tool IDs have been fully processed
        has_tool_calls = False

        # Process the stream
        if protocol == "text":
            # Simple text protocol - stream all text chunks as plain text
            async for chunk in output_stream:
                chunk_count += 1
                print(f"[DEBUG] Chunk {chunk_count}: {type(chunk).__name__}")

                if isinstance(chunk, dspy.streaming.StreamResponse):
                    # Stream any text output (reasoning, observations, answers)
                    print(
                        f"[DEBUG] StreamResponse field={chunk.signature_field_name}: {chunk.chunk}"
                    )
                    yield chunk.chunk

                elif isinstance(chunk, dspy.streaming.StatusMessage):
                    # For text protocol, we could optionally include status messages
                    # For now, we skip them to keep the output clean
                    print(
                        f"[DEBUG] StatusMessage (skipped in text mode): {chunk.message}"
                    )

                elif isinstance(chunk, dspy.Prediction):
                    print(f"[DEBUG] Final Prediction: {chunk}")
                    # Final output already streamed via chunks
                    pass

        elif protocol == "data":
            # Data protocol - SSE formatted stream with tool calls and metadata
            # Using AI SDK's SSE-based stream protocol
            message_id = str(uuid.uuid4())
            text_block_id = None

            async for chunk in output_stream:
                chunk_count += 1
                print(f"[DEBUG] Chunk {chunk_count}: {type(chunk).__name__}")

                if isinstance(chunk, dspy.streaming.StreamResponse):
                    # Stream text content using text-start/delta/end pattern
                    text_content = chunk.chunk
                    field_name = chunk.signature_field_name

                    print(f"[DEBUG] StreamResponse field={field_name}: {text_content}")

                    # Create a new text block ID if we don't have one
                    if text_block_id is None:
                        text_block_id = f"text_{uuid.uuid4().hex[:16]}"
                        # Emit text-start
                        output_line = (
                            f'data: {{"type":"text-start","id":"{text_block_id}"}}\n\n'
                        )
                        print(f"[DEBUG] Yielding text-start: {output_line.strip()}")
                        yield output_line

                    # Emit text-delta
                    output_line = f'data: {{"type":"text-delta","id":"{text_block_id}","delta":{json.dumps(text_content)}}}\n\n'
                    print(f"[DEBUG] Yielding text-delta: {output_line.strip()}")
                    yield output_line

                elif isinstance(chunk, dspy.streaming.StatusMessage):
                    # Handle tool call and LM status messages
                    message = chunk.message
                    print(f"[DEBUG] StatusMessage: {message}")

                    if message.startswith("tool_start:"):
                        # Parse: tool_start:<id>:<name>:<args_json>
                        parts = message.split(":", 3)
                        if len(parts) == 4:
                            tool_id = parts[1]
                            tool_name = parts[2]
                            tool_args_json = parts[3]

                            has_tool_calls = True

                            # Parse the args JSON
                            tool_args = json.loads(tool_args_json)

                            # Store pending tool call
                            tool_calls_pending[tool_id] = {
                                "name": tool_name,
                                "args": tool_args,
                            }

                            # Emit tool-input-start
                            output_line = f'data: {{"type":"tool-input-start","toolCallId":"{tool_id}","toolName":"{tool_name}"}}\n\n'
                            print(
                                f"[DEBUG] Yielding tool-input-start: {output_line.strip()}"
                            )
                            yield output_line

                            # Emit tool-input-available with the full input
                            output_line = f'data: {{"type":"tool-input-available","toolCallId":"{tool_id}","toolName":"{tool_name}","input":{json.dumps(tool_args)}}}\n\n'
                            print(
                                f"[DEBUG] Yielding tool-input-available: {output_line.strip()}"
                            )
                            yield output_line

                            # Emit a custom data part for reasoning/status
                            output_line = f'data: {{"type":"data-reasoning","data":{{"status":"calling_tool","toolName":"{tool_name}"}}}}\n\n'
                            print(f"[DEBUG] Yielding reasoning: {output_line.strip()}")
                            yield output_line

                    elif message.startswith("tool_end:"):
                        # Parse: tool_end:<id>:<result_json>
                        parts = message.split(":", 2)
                        if len(parts) == 3:
                            tool_id = parts[1]
                            tool_result_json = parts[2]

                            # Emit tool-output-available
                            if tool_id in tool_calls_pending:
                                tool_result = json.loads(tool_result_json)
                                output_line = f'data: {{"type":"tool-output-available","toolCallId":"{tool_id}","output":{json.dumps(tool_result)}}}\n\n'
                                print(
                                    f"[DEBUG] Yielding tool-output-available: {output_line.strip()}"
                                )
                                yield output_line

                                # Mark as completed
                                tool_calls_completed.add(tool_id)

                                # Emit reasoning update
                                output_line = f'data: {{"type":"data-reasoning","data":{{"status":"tool_complete","toolName":"{tool_calls_pending[tool_id]["name"]}"}}}}\n\n'
                                print(
                                    f"[DEBUG] Yielding reasoning complete: {output_line.strip()}"
                                )
                                yield output_line

                    elif message == "lm_start":
                        # Emit reasoning for LM start
                        output_line = f'data: {{"type":"data-reasoning","data":{{"status":"thinking"}}}}\n\n'
                        print(
                            f"[DEBUG] Yielding reasoning (thinking): {output_line.strip()}"
                        )
                        yield output_line

                    elif message == "lm_end":
                        # Emit reasoning for LM end
                        output_line = f'data: {{"type":"data-reasoning","data":{{"status":"done_thinking"}}}}\n\n'
                        print(
                            f"[DEBUG] Yielding reasoning (done): {output_line.strip()}"
                        )
                        yield output_line

                elif isinstance(chunk, dspy.Prediction):
                    # Final prediction - close text block and send finish message
                    print(f"[DEBUG] Final Prediction: {chunk}")

                    # Close the text block if we have one
                    if text_block_id is not None:
                        output_line = (
                            f'data: {{"type":"text-end","id":"{text_block_id}"}}\n\n'
                        )
                        print(f"[DEBUG] Yielding text-end: {output_line.strip()}")
                        yield output_line
                        text_block_id = None

                    # Emit finish message
                    output_line = f'data: {{"type":"finish"}}\n\n'
                    print(f"[DEBUG] Yielding finish: {output_line.strip()}")
                    yield output_line

            # Stream termination marker
            yield "data: [DONE]\n\n"

        print(f"[DEBUG] Stream completed. Total chunks: {chunk_count}")

    except Exception as e:
        print(f"[ERROR] Exception in stream_dspy_text: {e}")
        import traceback

        traceback.print_exc()

        # Send error message and finish
        if protocol == "text":
            yield f"Error: {str(e)}"
        else:
            # Emit error in SSE format
            error_msg = f"Error: {str(e)}"
            yield f'data: {{"type":"error","errorText":{json.dumps(error_msg)}}}\n\n'
            yield f'data: {{"type":"finish"}}\n\n'
            yield "data: [DONE]\n\n"


def stream_openai_text(messages: List[ClientMessage], protocol: str = "data"):
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location", "unit"],
                    },
                },
            }
        ],
    )

    # When protocol is set to "text", you will send a stream of plain text chunks
    # https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#text-stream-protocol

    if protocol == "text":
        for chunk in stream:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    break
                else:
                    yield "{text}".format(text=choice.delta.content)

    # When protocol is set to "data", you will send a stream data part chunks
    # https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol

    elif protocol == "data":
        draft_tool_calls = []
        draft_tool_calls_index = -1

        for chunk in stream:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    continue

                elif choice.finish_reason == "tool_calls":
                    for tool_call in draft_tool_calls:
                        yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                            id=tool_call["id"],
                            name=tool_call["name"],
                            args=tool_call["arguments"],
                        )

                    for tool_call in draft_tool_calls:
                        tool_result = available_tools[tool_call["name"]](
                            **json.loads(tool_call["arguments"])
                        )

                        yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                            id=tool_call["id"],
                            name=tool_call["name"],
                            args=tool_call["arguments"],
                            result=json.dumps(tool_result),
                        )

                elif choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        id = tool_call.id
                        name = tool_call.function.name
                        arguments = tool_call.function.arguments

                        if id is not None:
                            draft_tool_calls_index += 1
                            draft_tool_calls.append(
                                {"id": id, "name": name, "arguments": ""}
                            )

                        else:
                            draft_tool_calls[draft_tool_calls_index][
                                "arguments"
                            ] += arguments

                else:
                    yield "0:{text}\n".format(text=json.dumps(choice.delta.content))

            if chunk.choices == []:
                usage = chunk.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens

                yield 'd:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}}}}\n'.format(
                    reason="tool-calls" if len(draft_tool_calls) > 0 else "stop",
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                )


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    """
    Handle chat requests using DSPy ReAct with streaming.
    Supports both 'text' and 'data' protocols for Vercel AI SDK.
    """
    print("Request:", json.dumps(request.model_dump(), indent=2))
    print(f"Protocol: {protocol}")
    messages = request.messages

    # Use text/plain for text protocol, text/event-stream for data protocol (SSE)
    media_type = "text/plain" if protocol == "text" else "text/event-stream"

    response = StreamingResponse(
        stream_dspy_text(messages, protocol), media_type=media_type
    )

    # Set proper headers for SSE streaming
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering

    return response
