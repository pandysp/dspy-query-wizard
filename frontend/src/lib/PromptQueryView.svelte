<script lang="ts">
  import TokenizedTextView from "./TokenizedTextView.svelte";
  import type { AIQueryResult } from "./types";

  const {
    result,
    compareResult,
    title,
  }: { result: AIQueryResult; compareResult?: AIQueryResult; title: string } =
    $props();
</script>

<div class="p-2">
  <p
    class="uppercase tracking-wider text-center text-sm shadow drop-shadow-[0_0_8px_white]"
  >
    {title}
  </p>

  <div class="mt-4">
    <p class="text-sm text-gray-400 pl-2 font-mono">Input</p>
    <TokenizedTextView
      text={result.prompt}
      compareText={compareResult?.prompt}
    />

    <p class="text-sm text-gray-400 pl-2 font-mono">Agent Response</p>
    <TokenizedTextView
      text={result.llm_answer}
      compareText={compareResult?.llm_answer}
    />
  </div>

  <p class="text-sm text-gray-400 pl-2 font-mono">Score %</p>
  <p class="text-sm text-gray-400 pl-2 font-mono">{result.score}%</p>
</div>
