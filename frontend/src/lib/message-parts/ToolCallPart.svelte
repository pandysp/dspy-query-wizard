<script lang="ts">
  interface Props {
    toolName: string;
    input: unknown;
    state?: string;
  }

  let { toolName, input, state }: Props = $props();
</script>

<p class="text-xs text-gray-500 font-mono">tool</p>
<div>
  <div>{toolName}{state ? ` (${state})` : ""}</div>
  <pre>{JSON.stringify(input, null, 2)}</pre>
</div>
