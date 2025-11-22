<script lang="ts">
  import type { AgentPhase } from '../agent-response-processor';
  import { formatDuration } from '../agent-response-processor';

  interface Props {
    phase: AgentPhase;
  }

  let { phase }: Props = $props();

  let expanded = $state(false);

  const duration = $derived(formatDuration(phase.startTime, phase.endTime));
  const isStreaming = $derived(phase.thoughts?.isStreaming || false);
  const thoughtText = $derived(phase.thoughts?.text || '');

  // Auto-expand if streaming
  $effect(() => {
    if (isStreaming) {
      expanded = true;
    }
  });
</script>

<div class="phase-reasoning border-l-2 border-purple-600 pl-3 py-2 rounded-sm bg-black bg-opacity-30">
  <button
    class="w-full text-left"
    onclick={() => (expanded = !expanded)}
  >
    <div class="flex items-center justify-between">
      <div class="text-sm font-semibold text-purple-400">
        {#if isStreaming}
          💭 Thinking...
          <span class="inline-block animate-pulse ml-1">●</span>
        {:else}
          💭 Thought {duration ? `for ${duration}` : ''}
        {/if}
      </div>
      <div class="text-gray-500 text-xs">
        {expanded ? '▼' : '▶'}
      </div>
    </div>
  </button>

  {#if expanded && thoughtText}
    <div class="mt-2 text-xs text-gray-300 whitespace-pre-wrap">
      {thoughtText}
      {#if isStreaming}
        <span class="inline-block animate-pulse">▋</span>
      {/if}
    </div>
  {/if}
</div>
