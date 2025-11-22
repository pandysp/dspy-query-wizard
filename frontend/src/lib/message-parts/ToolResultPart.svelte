<script lang="ts">
  interface Props {
    toolName: string;
    output: unknown;
  }

  let { toolName, output }: Props = $props();
</script>

ToolResult
<div>
  <div>{toolName}</div>
  <pre>{JSON.stringify(output, null, 2)}</pre>
</div>
