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
  import { cn } from "./utils";
  import { inputPrompts, selectedInputPrompt } from "./configStore.svelte";

  let input = $state("");
  let isLoading = $state(false);

  const { systemMessagePrompt }: { systemMessagePrompt: string } = $props();

  const chat = new Chat({
    transport: new DefaultChatTransport({
      api: "http://127.0.0.1:8000/api/chat",
      prepareSendMessagesRequest: ({ id, messages }) => {
        return {
          body: {
            id,
            messages,
            system_message: systemMessagePrompt,
          },
        };
      },
    }),
    onFinish: (finish) => {
      isLoading = false;
    },
    onError: (error) => {
      isLoading = false;
    },
  });

  const showReasoning = $state(false);

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

  const getRoleEmoji = (role: string) => {
    switch (role) {
      case "user":
        return "👤";
      case "agent":
        return "🤖";
      case "system":
        return "⚙️";
    }
  };

  const fullMessages = $derived(() => {
    const backendMessages = chat.messages;
    const messages = [
      {
        role: "system",
        parts: [
          {
            type: "text",
            text: systemMessagePrompt,
          },
        ],
      },
      ...backendMessages.map((msg) => ({
        ...msg,
        parts: msg.parts.filter((part) => {
          // Always filter out incomplete tool calls as before
          const isIncompleteToolCall =
            part.type &&
            part.type.startsWith?.("tool-") &&
            "state" in part &&
            "input" in part &&
            part.input !== undefined &&
            part.state !== "output-available";

          // If showReasoning is false, filter out all reasoning parts
          if (!showReasoning && part.type === "reasoning") {
            return false;
          }

          return !isIncompleteToolCall;
        }),
      })),
    ];

    return messages;
  });

  // Start loading
  const inputPrompt = inputPrompts.prompts.find(
    (p) => p.id === selectedInputPrompt.id,
  )?.prompt;
  if (inputPrompt) {
    chat.sendMessage({ text: inputPrompt });
    isLoading = true;
  }
</script>

<div class="px-2">
  <div class="space-y-4 mb-4">
    {#each fullMessages() as message, messageIndex (messageIndex)}
      <div>
        <p
          class={cn(
            "uppercase text-sm font-mono w-full border-b border-dashed border-gray-600",
          )}
        >
          {getRoleEmoji(message.role)}
          {message.role}
        </p>

        <div
          class={cn("space-y-2 mt-2", message.role === "system" && "min-h-22")}
        >
          {#each message.parts as part, partIndex (partIndex)}
            <div class="rounded-sm bg-black text-gray-400 p-1 w-fit pl-2 pr-4">
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
                    typeof data.toolName === "string"
                      ? data.toolName
                      : undefined}

                  <ReasoningPart {status} {toolName} {data} />
                {/if}
              {:else}
                <!-- Unknown part types (for debugging) -->
                <UnknownPart type={part.type} data={part} />
              {/if}
            </div>
          {/each}
        </div>
      </div>
    {/each}

    {#if isLoading}
      <div class="text-center text-gray-500 py-4">
        <div class="animate-pulse">Agent is processing...</div>
      </div>
    {/if}
  </div>
</div>
