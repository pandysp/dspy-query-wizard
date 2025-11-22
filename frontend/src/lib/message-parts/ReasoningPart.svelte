<script lang="ts">
  interface Props {
    status: string;
    toolName?: string;
    data?: Record<string, unknown>;
  }

  let { status, toolName, data }: Props = $props();
</script>

<div>
  {#if status === "thinking"}
    <div>Thinking...</div>
  {:else if status === "done_thinking"}
    <div>Done thinking</div>
  {:else if status === "calling_tool"}
    <div>Calling tool: {toolName}</div>
  {:else if status === "tool_complete"}
    <div>Tool complete: {toolName}</div>
  {:else if data}
    <pre>{JSON.stringify(data, null, 2)}</pre>
  {:else}
    <div>Status: {status}</div>
  {/if}
</div>
