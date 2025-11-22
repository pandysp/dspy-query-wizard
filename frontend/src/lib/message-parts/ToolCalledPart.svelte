<script lang="ts">
  interface Props {
    toolName: string;
    input: unknown;
  }

  let { toolName, input }: Props = $props();
</script>

ToolCalled
<div>
  <div>{toolName}</div>
  <pre>{JSON.stringify(input, null, 2)}</pre>
</div>
