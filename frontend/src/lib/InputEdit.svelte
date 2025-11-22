<script lang="ts">
  import {
    inputPrompts,
    selectedInputPrompt,
    sytemMessages,
  } from "./configStore.svelte";

  const { onRunClicked } = $props();
</script>

<section class="flex justify-center items-center min-h-screen">
  <div class="flex flex-col gap-2">
    <p class="text-gray-400">Enter a system message</p>
    <textarea
      bind:value={sytemMessages.default}
      class="w-lg h-48 p-2 pb-8 rounded-md bg-gray-800 text-white resize-none shadow-inners relative"
    ></textarea>
    <p class="text-gray-400">Select example user input</p>
    <select
      bind:value={selectedInputPrompt.id}
      class="w-lg h-10 p-2 rounded-md bg-gray-800 text-white resize-none shadow-inners relative"
    >
      {#each inputPrompts.prompts as prompt}
        <option value={prompt.id}>{prompt.prompt}</option>
      {/each}
    </select>

    <button
      class="rounded mt-2 p-2 bg-green-600 hover:bg-green-700 text-white cursor-pointer"
      onclick={onRunClicked}>Evaluate</button
    >
  </div>
</section>
