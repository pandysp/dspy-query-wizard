<script lang="ts">
  import type { AgentPhase } from '../agent-response-processor';
  import { formatDuration } from '../agent-response-processor';
  import ToolCallCard from './ToolCallCard.svelte';

  interface Props {
    phase: AgentPhase;
  }

  let { phase }: Props = $props();

  let expanded = $state(false);

  const duration = $derived(formatDuration(phase.startTime, phase.endTime));
  const toolCount = $derived(phase.toolCalls?.length || 0);
  const isComplete = $derived(!!phase.endTime);
</script>

<div class="phase-research border-l-2 border-gray-700 pl-3 py-2 rounded-sm bg-black bg-opacity-30">
  <button
    class="w-full text-left"
    onclick={() => (expanded = !expanded)}
  >
    <div class="flex items-center justify-between">
      <div class="text-sm font-semibold text-gray-300">
        {#if isComplete}
          🔍 Research {duration ? `(${duration})` : ''} - {toolCount} tool{toolCount !== 1 ? 's' : ''}
        {:else}
          🔍 Searching... ({toolCount} tool{toolCount !== 1 ? 's' : ''} active)
        {/if}
      </div>
      <div class="text-gray-500 text-xs">
        {expanded ? '▼' : '▶'}
      </div>
    </div>
  </button>

  {#if expanded && phase.toolCalls && phase.toolCalls.length > 0}
    <div class="mt-2 space-y-2">
      {#each phase.toolCalls as toolCall (toolCall.id)}
        <ToolCallCard {toolCall} />
      {/each}
    </div>
  {/if}
</div>
