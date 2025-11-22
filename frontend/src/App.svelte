<script lang="ts">
  import AgentChat from "./lib/AgentChat.svelte";
  import PromptQueryView from "./lib/PromptQueryView.svelte";
  import type { AIQueryResult } from "./lib/types";

  const dummyHuman: AIQueryResult = {
    prompt: "What is the capital of France?",
    llm_answer: "Paris.",
    score: 50,
    chunks: [],
  };

  const dummyAI: AIQueryResult = {
    prompt:
      "What is the capital of France? But rephrased to be more intelligent.",
    llm_answer: "Paris, your majesty.",
    score: 75,
    chunks: [],
  };
</script>

<main class="bg-gray-900 text-white min-h-screen">
  <div class="grid grid-cols-2 w-full gap-0 pt-12">
    <AgentChat />

    <div
      class="absolute translate-x-[-50%] inset-y-4 left-1/2 w-px bg-gray-700"
    ></div>
  </div>
</main>
