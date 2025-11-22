<script lang="ts">
  import { parseWikipediaResult } from '../agent-response-processor';

  interface Props {
    item: string | unknown;
  }

  let { item }: Props = $props();

  let showFull = $state(false);
  const parsed = $derived(parseWikipediaResult(item));
</script>

<div class="wiki-result border-l-2 border-green-600 pl-3 py-1">
  <div class="text-sm font-semibold text-gray-300">📄 {parsed.title}</div>
  <div class="text-xs text-gray-400 mt-1">
    {showFull ? parsed.fullContent : parsed.snippet}
  </div>
  {#if parsed.snippet !== parsed.fullContent}
    <button
      class="text-xs text-blue-400 hover:text-blue-300 mt-1 underline"
      onclick={() => (showFull = !showFull)}
    >
      {showFull ? 'Show less' : 'Read more'}
    </button>
  {/if}
</div>
