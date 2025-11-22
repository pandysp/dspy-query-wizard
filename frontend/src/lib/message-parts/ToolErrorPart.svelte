<script lang="ts">
  interface Props {
    toolName: string;
    errorText?: string;
  }

  let { toolName, errorText }: Props = $props();
</script>

<div class="bg-red-100 border border-red-300 rounded p-2">
  <div class="text-xs font-mono text-red-800 mb-1">❌ TOOL ERROR</div>
  <div class="font-semibold text-sm">{toolName}</div>
  {#if errorText}
    <div class="text-sm text-red-900">{errorText}</div>
  {/if}
</div>
