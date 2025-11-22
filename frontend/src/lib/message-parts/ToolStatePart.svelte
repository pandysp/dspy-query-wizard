<script lang="ts">
  interface Props {
    toolName: string;
    state: string;
  }

  let { toolName, state }: Props = $props();
</script>

<div>
  <div>{toolName}</div>
  <div>State: {state}</div>
</div>
