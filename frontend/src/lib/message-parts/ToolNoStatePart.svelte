<script lang="ts">
  interface Props {
    type: string;
    data: unknown;
  }

  let { type, data }: Props = $props();
</script>

<div class="bg-orange-100 border border-orange-300 rounded p-2">
  <div class="text-xs font-mono mb-1">🔧 {type} (no state)</div>
  <pre class="text-xs">{JSON.stringify(data, null, 2)}</pre>
</div>
