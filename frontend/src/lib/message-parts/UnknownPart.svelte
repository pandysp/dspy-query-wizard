<script lang="ts">
  interface Props {
    type: string;
    data: unknown;
  }

  let { type, data }: Props = $props();
</script>

<div class="bg-gray-200 border border-gray-400 rounded p-2">
  <div class="text-xs font-mono mb-1">❓ {type}</div>
  <pre class="text-xs overflow-x-auto">{JSON.stringify(data, null, 2)}</pre>
</div>
