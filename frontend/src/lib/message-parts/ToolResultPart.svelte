<script lang="ts">
  interface Props {
    toolName: string;
    output: unknown;
  }

  let { toolName, output }: Props = $props();
</script>

<p class="text-xs text-gray-500 font-mono">tool result</p>
<div>
  <div>{toolName}</div>
  <pre>{JSON.stringify(output, null, 2)}</pre>
</div>
