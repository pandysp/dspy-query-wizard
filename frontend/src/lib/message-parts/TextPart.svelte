<script lang="ts">
  interface Props {
    text: string;
  }

  let { text }: Props = $props();
</script>

<p class="text-xs text-gray-500 font-mono">text</p>
<div
  class="max-w-xs break-after-all line-clamp-3 hover:line-clamp-none hover:whitespace-normal"
  title={text}
>
  {text}
</div>
