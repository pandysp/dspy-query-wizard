<script lang="ts">
  const { role }: { role: "user" | "agent" | "system" } = $props();
</script>

<div>
  <div class="font-semibold text-sm uppercase mb-2">{role}</div>
</div>
