<script lang="ts">
  interface Props {
    status: string;
    toolName?: string;
    data?: Record<string, unknown>;
  }

  let { status, toolName, data }: Props = $props();
</script>

<div class="bg-purple-100 border border-purple-300 rounded p-2">
  <div class="text-xs font-mono text-purple-800 mb-1">🧠 REASONING</div>
  {#if status === "thinking"}
    <div class="text-sm">💭 Thinking...</div>
  {:else if status === "done_thinking"}
    <div class="text-sm">✓ Done thinking</div>
  {:else if status === "calling_tool"}
    <div class="text-sm">⚙️ Calling tool: {toolName}</div>
  {:else if status === "tool_complete"}
    <div class="text-sm">✓ Tool complete: {toolName}</div>
  {:else if data}
    <pre class="text-xs">{JSON.stringify(data, null, 2)}</pre>
  {:else}
    <div class="text-sm">Status: {status}</div>
  {/if}
</div>
