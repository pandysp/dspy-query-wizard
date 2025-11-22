<script lang="ts">
  interface Props {
    toolName: string;
    input: unknown;
    state?: string;
  }

  let { toolName, input, state }: Props = $props();
</script>

<div class="bg-yellow-100 border border-yellow-300 rounded p-2">
  <div class="text-xs font-mono text-yellow-800 mb-1">
    🔧 TOOL CALL {state ? `(${state})` : ""}
  </div>
  <div class="font-semibold text-sm">{toolName}</div>
  <pre class="text-xs mt-1 overflow-x-auto">{JSON.stringify(
      input,
      null,
      2,
    )}</pre>
</div>
