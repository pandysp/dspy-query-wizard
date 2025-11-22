<script lang="ts">
  interface Props {
    toolName: string;
    input: unknown;
  }

  let { toolName, input }: Props = $props();
</script>

<p class="text-xs text-gray-500 font-mono">tool</p>
<div>
  <div>{toolName}</div>
  <pre>{JSON.stringify(input, null, 2)}</pre>
</div>
