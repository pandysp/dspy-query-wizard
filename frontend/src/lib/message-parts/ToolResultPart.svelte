<script lang="ts">
  interface Props {
    toolName: string;
    output: unknown;
  }

  let { toolName, output }: Props = $props();
</script>

<div class="bg-green-100 border border-green-300 rounded p-2">
  <div class="text-xs font-mono text-green-800 mb-1">✅ TOOL RESULT</div>
  <div class="font-semibold text-sm">{toolName}</div>
  <pre class="text-xs mt-1 overflow-x-auto">{JSON.stringify(
      output,
      null,
      2,
    )}</pre>
</div>
