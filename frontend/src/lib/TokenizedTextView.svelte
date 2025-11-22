<script lang="ts">
  interface Props {
    text: string;
    compareText?: string;
  }

  import { estimateTokenCount } from "tokenx";
  const { text, compareText }: Props = $props();
  const tokenCount = $derived(estimateTokenCount(text));
  const compareTokenCount = $derived(
    compareText ? estimateTokenCount(compareText) : 0,
  );

  const tokenCountString = $derived(
    `${tokenCount} ${compareTokenCount > 0 ? `(+${compareTokenCount})` : ""}`,
  );
</script>

<div class="relative">
  <textarea
    readonly
    value={text}
    class="w-full p-2 pb-8 rounded-md h-30 bg-gray-800 text-white resize-none shadow-inners relative"
  ></textarea>
  <div class="absolute bottom-3 right-2 text-xs text-gray-400">
    {tokenCountString} tokens
  </div>
</div>
