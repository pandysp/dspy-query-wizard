<script lang="ts">
  import { Chat } from "@ai-sdk/svelte";
  import { DefaultChatTransport } from "ai";
  import { AgentResponseProcessor, type AgentPhase, type StreamEvent } from "./agent-response-processor";
  import { ResearchPhase, ReasoningPhase, AnswerPhase } from "./phase-components";
  import { TextPart } from "./message-parts";
  import { cn } from "./utils";
  import { inputPrompts, selectedInputPrompt } from "./configStore.svelte";

  let isLoading = $state(false);
  let evaluation = $state<number | null>(null);
  const EVALUATE_TOOL = "evaluate";

  const {
    systemMessagePrompt,
    isOptimized,
  }: { systemMessagePrompt: string; isOptimized?: boolean } = $props();

  const chat = new Chat({
    transport: new DefaultChatTransport({
      api: "http://127.0.0.1:8000/api/chat",
      prepareSendMessagesRequest: ({ id, messages, trigger }) => {
        const body = {
          id,
          messages,
          trigger,
        } as {
          id: string;
          messages: typeof messages;
          system_message?: string;
          trigger: typeof trigger;
        };
        if (!isOptimized) {
          body.system_message = systemMessagePrompt;
        }
        return { body };
      },
    }),
    onFinish: (finish) => {
      isLoading = false;
    },
    onError: (error) => {
      console.error("❌ Stream error:", error);
      isLoading = false;
    },
  });

  const getRoleEmoji = (role: string) => {
    switch (role) {
      case "user":
        return "👤";
      case "assistant":
        return "🤖";
      case "system":
        return "⚙️";
      default:
        return "🔹";
    }
  };

  // Transform message parts into phases for agent messages
  function transformPartsToPhases(parts: any[]): AgentPhase[] {
    console.log("🔄 Transforming parts:", parts.map((p: any) => p.type));
    const processor = new AgentResponseProcessor();

    for (const part of parts) {
      // Handle tool calls (type starts with "tool-")
      if (part.type && part.type.startsWith("tool-")) {
        const toolName = part.type.replace("tool-", "");

        // Tool input start
        processor.processEvent({
          type: "tool-input-start",
          toolCallId: part.toolCallId || crypto.randomUUID(),
          toolName: toolName,
        } as StreamEvent);

        // Tool input available
        if ("input" in part) {
          processor.processEvent({
            type: "tool-input-available",
            toolCallId: part.toolCallId || crypto.randomUUID(),
            toolName: toolName,
            input: part.input as Record<string, unknown>,
          } as StreamEvent);
        }

        // Tool output
        if ("state" in part && part.state === "output-available" && "output" in part) {
          processor.processEvent({
            type: "tool-output-available",
            toolCallId: part.toolCallId || crypto.randomUUID(),
            output: part.output,
          } as StreamEvent);
        }
      }
      // Handle reasoning (aggregated by AI SDK)
      else if (part.type === "reasoning") {
        processor.processEvent({
          type: "reasoning-start",
          id: crypto.randomUUID(),
        } as StreamEvent);

        if ("text" in part && part.text) {
          processor.processEvent({
            type: "reasoning-delta",
            delta: part.text,
          } as StreamEvent);
        }

        processor.processEvent({
          type: "reasoning-end",
        } as StreamEvent);
      }
      // Handle text
      else if (part.type === "text") {
        processor.processEvent({
          type: "text-start",
          id: crypto.randomUUID(),
        } as StreamEvent);

        if ("text" in part && part.text) {
          processor.processEvent({
            type: "text-delta",
            delta: part.text,
          } as StreamEvent);
        }

        processor.processEvent({
          type: "text-end",
        } as StreamEvent);
      }
    }

    processor.processEvent({ type: "finish" } as StreamEvent);
    const phases = processor.getPhases();
    console.log("✅ Generated phases:", phases.map(p => `${p.type}(${p.toolCalls?.length || 0} tools, ${p.thoughts ? 'thoughts' : 'no thoughts'}, ${p.answer ? 'answer' : 'no answer'})`));
    return phases;
  }

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
        phases: [] as AgentPhase[],
      },
      ...backendMessages.map((msg) => ({
        ...msg,
        // Keep original parts for system/user messages, transform for assistant
        phases: msg.role === "assistant" ? transformPartsToPhases(msg.parts) : [] as AgentPhase[],
      })),
    ];

    return messages;
  });

  $effect(() => {
    // Find first tool result for EVALUATE_TOOL
    for (const message of chat.messages) {
      for (const part of message.parts) {
        if (
          part.type === `tool-${EVALUATE_TOOL}` &&
          "state" in part &&
          part.state === "output-available" &&
          "output" in part
        ) {
          if (typeof part.output === "number") {
            evaluation = part.output;
          } else {
            evaluation = parseFloat(part.output as string);
          }
          return;
        }
      }
    }
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
    {#if evaluation !== null}
      <div class="text-center text-gray-500 py-4">
        <div class="text-sm">Evaluation:</div>
        <div class="text-xl">{evaluation * 100}%</div>
      </div>
    {/if}
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
          {#if message.role === "assistant" && message.phases && message.phases.length > 0}
            <!-- Render agent messages using phases -->
            {#each message.phases as phase (phase.id)}
              {#if phase.type === "research"}
                <ResearchPhase {phase} />
              {:else if phase.type === "reasoning"}
                <ReasoningPhase {phase} />
              {:else if phase.type === "answer"}
                <AnswerPhase {phase} />
              {/if}
            {/each}
          {:else}
            <!-- Render system/user messages using simple text parts -->
            {#each message.parts as part, partIndex (partIndex)}
              {#if part.type === "text"}
                <div class="rounded-sm bg-black text-gray-400 p-1 w-fit pl-2 pr-4">
                  <TextPart text={part.text} />
                </div>
              {/if}
            {/each}
          {/if}
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
