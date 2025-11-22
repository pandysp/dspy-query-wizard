<script lang="ts">
  import type { AgentPhase } from '../agent-response-processor';

  interface Props {
    phase: AgentPhase;
  }

  let { phase }: Props = $props();

  const isStreaming = $derived(phase.answer?.isStreaming || false);
  const answerText = $derived(phase.answer?.text || '');
</script>

<div class="phase-answer border-l-2 border-green-600 pl-3 py-2 rounded-sm bg-black bg-opacity-30">
  <div class="text-sm font-semibold text-green-400 mb-2">✍️ Answer</div>
  <div class="text-sm text-gray-200 whitespace-pre-wrap">
    {answerText}
    {#if isStreaming}
      <span class="inline-block animate-pulse">▋</span>
    {/if}
  </div>
</div>
