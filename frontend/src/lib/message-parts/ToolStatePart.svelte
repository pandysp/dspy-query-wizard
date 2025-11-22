<script lang="ts">
  interface Props {
    toolName: string;
    state: string;
  }

  let { toolName, state }: Props = $props();
</script>

<div class="bg-gray-100 border border-gray-300 rounded p-2">
  <div class="text-xs font-mono mb-1">🔧 {toolName}</div>
  <div class="text-xs text-gray-600">State: {state}</div>
</div>
