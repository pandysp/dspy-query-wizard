<script lang="ts">
  interface Props {
    toolName: string;
    input: unknown;
  }

  let { toolName, input }: Props = $props();
</script>

<div class="bg-yellow-50 border border-yellow-200 rounded p-2">
  <div class="text-xs font-mono text-yellow-700 mb-1">🔧 TOOL CALLED</div>
  <div class="font-semibold text-sm">{toolName}</div>
  <pre class="text-xs mt-1 overflow-x-auto">{JSON.stringify(
      input,
      null,
      2,
    )}</pre>
</div>
