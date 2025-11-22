<script lang="ts">
  interface Props {
    toolName: string;
    errorText?: string;
  }

  let { toolName, errorText }: Props = $props();
</script>

<div>
  <div>{toolName}</div>
  {#if errorText}
    <div>{errorText}</div>
  {/if}
</div>
