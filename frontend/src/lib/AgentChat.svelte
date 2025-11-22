<script lang="ts">
  import { Chat } from "@ai-sdk/svelte";
  import { DefaultChatTransport } from "ai";

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
    onToolCall: (toolCall) => {
      console.log("🔧 toolCall", toolCall);
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
            <!-- Text content -->
            {#if part.type === "text"}
              <div class="text-gray-900">{part.text}</div>

              <!-- Tool Call (when tool is invoked) -->
            {:else if part.type.startsWith("tool-")}
              {#if "state" in part}
                {#if part.state === "input-streaming" || part.state === "input-available"}
                  <div
                    class="bg-yellow-100 border border-yellow-300 rounded p-2"
                  >
                    <div class="text-xs font-mono text-yellow-800 mb-1">
                      🔧 TOOL CALL
                    </div>
                    <div class="font-semibold text-sm">
                      {part.type.replace("tool-", "")}
                    </div>
                    {#if "input" in part && part.input !== undefined}
                      <pre class="text-xs mt-1 overflow-x-auto">{JSON.stringify(
                          part.input,
                          null,
                          2,
                        )}</pre>
                    {/if}
                  </div>
                {:else if part.state === "output-available"}
                  <div class="bg-green-100 border border-green-300 rounded p-2">
                    <div class="text-xs font-mono text-green-800 mb-1">
                      ✅ TOOL RESULT
                    </div>
                    <div class="font-semibold text-sm">
                      {part.type.replace("tool-", "")}
                    </div>
                    {#if "output" in part}
                      <pre class="text-xs mt-1 overflow-x-auto">{JSON.stringify(
                          part.output,
                          null,
                          2,
                        )}</pre>
                    {/if}
                  </div>
                {:else if part.state === "output-error"}
                  <div class="bg-red-100 border border-red-300 rounded p-2">
                    <div class="text-xs font-mono text-red-800 mb-1">
                      ❌ TOOL ERROR
                    </div>
                    <div class="font-semibold text-sm">
                      {part.type.replace("tool-", "")}
                    </div>
                    {#if "errorText" in part && part.errorText}
                      <div class="text-sm text-red-900">{part.errorText}</div>
                    {/if}
                  </div>
                {:else}
                  <!-- Other tool states (streaming, done, etc) -->
                  <div class="bg-gray-100 border border-gray-300 rounded p-2">
                    <div class="text-xs font-mono mb-1">
                      🔧 {part.type.replace("tool-", "")}
                    </div>
                    <div class="text-xs text-gray-600">State: {part.state}</div>
                  </div>
                {/if}
              {/if}

              <!-- Custom Reasoning Data -->
            {:else if part.type === "data-reasoning"}
              <div class="bg-purple-100 border border-purple-300 rounded p-2">
                <div class="text-xs font-mono text-purple-800 mb-1">
                  🧠 REASONING
                </div>
                {#if "data" in part && part.data && typeof part.data === "object"}
                  {@const data = part.data as Record<string, unknown>}
                  {#if data.status === "thinking"}
                    <div class="text-sm">💭 Thinking...</div>
                  {:else if data.status === "done_thinking"}
                    <div class="text-sm">✓ Done thinking</div>
                  {:else if data.status === "calling_tool"}
                    <div class="text-sm">⚙️ Calling tool: {data.toolName}</div>
                  {:else if data.status === "tool_complete"}
                    <div class="text-sm">✓ Tool complete: {data.toolName}</div>
                  {:else}
                    <pre class="text-xs">{JSON.stringify(data, null, 2)}</pre>
                  {/if}
                {/if}
              </div>

              <!-- Unknown part types (for debugging) -->
            {:else}
              <div class="bg-gray-200 border border-gray-400 rounded p-2">
                <div class="text-xs font-mono mb-1">❓ {part.type}</div>
                <pre class="text-xs overflow-x-auto">{JSON.stringify(
                    part,
                    null,
                    2,
                  )}</pre>
              </div>
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
