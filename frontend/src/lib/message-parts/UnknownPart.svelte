<script lang="ts">
  interface Props {
    type: string;
    data: unknown;
  }

  let { type, data }: Props = $props();
</script>

<div>
  <div>{type}</div>
  <pre>{JSON.stringify(data, null, 2)}</pre>
</div>
