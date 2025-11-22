<script lang="ts">
  interface Props {
    toolName: string;
    input: unknown;
    state?: string;
  }

  let { toolName, input, state }: Props = $props();
</script>

ToolCall
<div>
  <div>{toolName}{state ? ` (${state})` : ""}</div>
  <pre>{JSON.stringify(input, null, 2)}</pre>
</div>
