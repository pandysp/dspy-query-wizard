<script lang="ts">
  import type { ToolCall } from '../agent-response-processor';
  import WikipediaResult from './WikipediaResult.svelte';

  interface Props {
    toolCall: ToolCall;
  }

  let { toolCall }: Props = $props();

  let expanded = $state(false);

  const formatInput = (input?: Record<string, unknown>): string => {
    if (!input) return '';
    const parts: string[] = [];

    // Handle common patterns
    if ('query' in input && typeof input.query === 'string') {
      parts.push(`Query: "${input.query}"`);
    } else if ('kwargs' in input && typeof input.kwargs === 'object' && input.kwargs !== null) {
      const kwargs = input.kwargs as Record<string, unknown>;
      if ('query' in kwargs && typeof kwargs.query === 'string') {
        parts.push(`Query: "${kwargs.query}"`);
      }
      if ('k' in kwargs) {
        parts.push(`k: ${kwargs.k}`);
      }
    }

    // Fallback to all keys
    if (parts.length === 0) {
      for (const [key, value] of Object.entries(input)) {
        if (key !== 'kwargs') {
          parts.push(`${key}: ${JSON.stringify(value)}`);
        }
      }
    }

    return parts.join(', ');
  };

  const isWikipediaResults = $derived(
    toolCall.name === 'search_wikipedia' &&
    Array.isArray(toolCall.output)
  );
</script>

<div class="tool-call border-l-2 border-blue-600 pl-3 py-2">
  <button
    class="w-full text-left"
    onclick={() => (expanded = !expanded)}
  >
    <div class="flex items-center justify-between">
      <div>
        <div class="text-sm font-semibold text-blue-400">
          🔧 {toolCall.name}
        </div>
        <div class="text-xs text-gray-400 mt-1">
          {formatInput(toolCall.input)}
        </div>
      </div>
      <div class="text-gray-500 text-xs ml-2">
        {#if toolCall.status === 'pending'}
          ⏳
        {:else if toolCall.status === 'complete'}
          {expanded ? '▼' : '▶'}
        {:else if toolCall.status === 'error'}
          ❌
        {/if}
      </div>
    </div>
  </button>

  {#if expanded && toolCall.status === 'complete'}
    <div class="mt-2 space-y-2">
      {#if isWikipediaResults && Array.isArray(toolCall.output)}
        <div class="text-xs text-gray-500 mb-2">
          Results ({toolCall.output.length}):
        </div>
        {#each toolCall.output as result}
          <WikipediaResult item={result} />
        {/each}
      {:else}
        <div class="text-xs text-gray-400">
          <pre class="overflow-x-auto">{JSON.stringify(toolCall.output, null, 2)}</pre>
        </div>
      {/if}
    </div>
  {/if}

  {#if toolCall.status === 'error' && toolCall.errorText}
    <div class="text-xs text-red-400 mt-2">
      Error: {toolCall.errorText}
    </div>
  {/if}
</div>
