<script lang="ts">
  import AgentChat from "./lib/AgentChat.svelte";
  import { getSystemMessages } from "./lib/api";
  import InputEdit from "./lib/InputEdit.svelte";
  import {
    inputPrompts,
    loadInputPrompts,
    loadSystemMessages,
    sytemMessages,
  } from "./lib/configStore.svelte";

  let running = $state(false);

  loadSystemMessages();
  loadInputPrompts();

  const loading = $derived(
    inputPrompts.prompts.length === 0 || sytemMessages.default === "",
  );
</script>

<main class="bg-gray-900 text-gray-400 min-h-screen">
  {#if !running}
    {#if !loading}
      <InputEdit
        onRunClicked={() => {
          running = true;
        }}
      />
    {/if}
  {:else}
    <div class="relative grid grid-cols-2 w-full gap-0 pt-12">
      <AgentChat systemMessagePrompt={sytemMessages.default} />
      <AgentChat systemMessagePrompt={sytemMessages.optimized} isOptimized />

      <div
        class="absolute translate-x-[-50%] top-4 bottom-4 left-1/2 w-px bg-gray-700"
      ></div>
    </div>
  {/if}
</main>
