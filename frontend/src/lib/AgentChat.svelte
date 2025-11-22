<script lang="ts">
  import { Chat } from "@ai-sdk/svelte";
  import { DefaultChatTransport } from "ai";

  // Import message part components
  import {
    TextPart,
    ToolCallPart,
    ToolCalledPart,
    ToolResultPart,
    ToolErrorPart,
    ToolStatePart,
    ToolNoStatePart,
    ReasoningPart,
    UnknownPart,
  } from "./message-parts";

  let input = $state("");
  let isLoading = $state(false);

  const chat = new Chat({
    transport: new DefaultChatTransport({
      api: "http://127.0.0.1:8000/api/chat",
    }),
    onFinish: (finish) => {
      console.log("✅ finish", finish);
      isLoading = false;
    },
    onData: (data) => {
      console.log("📦 data", data);
    },
    onToolCall: ({ toolCall }) => {
      console.log("🔧 toolCall", {
        toolCallId: toolCall.toolCallId,
        state: "state" in toolCall ? toolCall.state : "no-state",
        hasInput: "input" in toolCall,
        hasOutput: "output" in toolCall,
        fullToolCall: toolCall,
      });
    },
    onError: (error) => {
      console.error("❌ error", error);
      isLoading = false;
    },
  });

  function handleSubmit(event: SubmitEvent) {
    event.preventDefault();
    if (input.trim() && !isLoading) {
      isLoading = true;
      chat.sendMessage({ text: input });
      input = "";
    }
  }

  // Debug: Log messages as they update
  $effect(() => {
    if (chat.messages.length > 0) {
      const lastMessage = chat.messages[chat.messages.length - 1];
      console.log(
        "📨 Last message parts:",
        lastMessage.parts.map((p) => ({
          type: p.type,
          state: "state" in p ? p.state : undefined,
          hasInput: "input" in p,
          hasOutput: "output" in p,
        })),
      );
    }
  });
</script>

<main class="max-w-2xl mx-auto p-4">
  <h1 class="text-2xl font-bold mb-4">DSPy ReAct Agent</h1>

  <div class="space-y-4 mb-4">
    {#each chat.messages as message, messageIndex (messageIndex)}
      <div
        class="border rounded-lg p-3 {message.role === 'user'
          ? 'bg-blue-50'
          : 'bg-gray-50'}"
      >
        <div class="font-semibold text-sm uppercase mb-2">
          {message.role === "user" ? "👤 User" : "🤖 Agent"}
        </div>

        <div class="space-y-2">
          {#each message.parts as part, partIndex (partIndex)}
            {#if part.type === "text"}
              <TextPart text={part.text} />
            {:else if part.type.startsWith("tool-")}
              {@const toolName = part.type.replace("tool-", "")}

              {#if "state" in part}
                <!-- Show tool call info when we have input (not yet completed) -->
                {#if "input" in part && part.input !== undefined && part.state !== "output-available"}
                  <ToolCallPart
                    {toolName}
                    input={part.input}
                    state={part.state}
                  />
                {/if}

                <!-- Show tool result when available -->
                {#if part.state === "output-available"}
                  <!-- Show input first -->
                  {#if "input" in part && part.input !== undefined}
                    <ToolCalledPart {toolName} input={part.input} />
                  {/if}

                  <!-- Then show output -->
                  {#if "output" in part}
                    <ToolResultPart {toolName} output={part.output} />
                  {/if}
                {:else if part.state === "output-error"}
                  <ToolErrorPart
                    {toolName}
                    errorText={"errorText" in part && part.errorText
                      ? String(part.errorText)
                      : undefined}
                  />
                {:else if part.state === "streaming" || part.state === "done"}
                  <ToolStatePart {toolName} state={part.state} />
                {/if}
              {:else}
                <!-- No state property - show debug -->
                <ToolNoStatePart type={part.type} data={part} />
              {/if}
            {:else if part.type === "data-reasoning"}
              {#if "data" in part && part.data && typeof part.data === "object"}
                {@const data = part.data as Record<string, unknown>}
                {@const status =
                  typeof data.status === "string" ? data.status : "unknown"}
                {@const toolName =
                  typeof data.toolName === "string" ? data.toolName : undefined}

                <ReasoningPart {status} {toolName} {data} />
              {/if}
            {:else}
              <!-- Unknown part types (for debugging) -->
              <UnknownPart type={part.type} data={part} />
            {/if}
          {/each}
        </div>
      </div>
    {/each}

    {#if isLoading}
      <div class="text-center text-gray-500 py-4">
        <div class="animate-pulse">⏳ Agent is processing...</div>
      </div>
    {/if}
  </div>

  <form onsubmit={handleSubmit} class="flex gap-2">
    <input
      bind:value={input}
      placeholder="Ask the agent something... (try: 'What's the weather in SF?')"
      class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      disabled={isLoading}
    />
    <button
      type="submit"
      class="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
      disabled={isLoading || !input.trim()}
    >
      Send
    </button>
  </form>
</main>
