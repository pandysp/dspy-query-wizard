# Message Part Components

This directory contains modular Svelte 5 components for rendering different types of message parts in the DSPy ReAct Agent chat interface.

## Component Architecture

Each component is focused on a single responsibility and receives only the props it needs. String manipulations (like extracting tool names from type strings) are performed in the parent component before passing data to children.

## Components

### 1. **TextPart.svelte**
Displays regular text content from messages.

**Props:**
- `text: string` - The text content to display

**Use case:** Standard text responses from the agent

---

### 2. **ToolCallPart.svelte**
Shows a tool being invoked with its input parameters (active/in-progress state).

**Props:**
- `toolName: string` - Name of the tool (e.g., "get_current_weather")
- `input: unknown` - Input parameters passed to the tool
- `state?: string` - Optional state indicator (e.g., "input-available")

**Use case:** When a tool is being called but hasn't completed yet

**Visual:** Yellow box with 🔧 icon

---

### 3. **ToolCalledPart.svelte**
Shows that a tool was called (past tense, lighter styling).

**Props:**
- `toolName: string` - Name of the tool
- `input: unknown` - Input parameters that were passed

**Use case:** Displayed alongside ToolResultPart to show both input and output

**Visual:** Light yellow box with 🔧 icon

---

### 4. **ToolResultPart.svelte**
Displays the result/output from a completed tool execution.

**Props:**
- `toolName: string` - Name of the tool
- `output: unknown` - The result returned by the tool

**Use case:** When a tool has successfully completed execution

**Visual:** Green box with ✅ icon

---

### 5. **ToolErrorPart.svelte**
Shows errors that occurred during tool execution.

**Props:**
- `toolName: string` - Name of the tool that errored
- `errorText?: string` - Optional error message

**Use case:** When a tool execution fails

**Visual:** Red box with ❌ icon

---

### 6. **ToolStatePart.svelte**
Displays other tool states (streaming, done, etc.).

**Props:**
- `toolName: string` - Name of the tool
- `state: string` - The current state

**Use case:** For intermediate tool states during execution

**Visual:** Gray box with 🔧 icon and state label

---

### 7. **ToolNoStatePart.svelte**
Debug component for tool parts that don't have a state property.

**Props:**
- `type: string` - The raw type string
- `data: unknown` - The full data object for debugging

**Use case:** Debugging malformed tool messages

**Visual:** Orange box with 🔧 icon and "(no state)" indicator

---

### 8. **ReasoningPart.svelte**
Shows the agent's reasoning and thought process.

**Props:**
- `status: string` - The reasoning status (e.g., "thinking", "calling_tool")
- `toolName?: string` - Optional tool name if relevant
- `data?: Record<string, unknown>` - Full data object for fallback display

**Use case:** Real-time insight into agent's decision-making process

**States handled:**
- `thinking` - 💭 Thinking...
- `done_thinking` - ✓ Done thinking
- `calling_tool` - ⚙️ Calling tool: [name]
- `tool_complete` - ✓ Tool complete: [name]

**Visual:** Purple box with 🧠 icon

---

### 9. **UnknownPart.svelte**
Fallback component for unknown/unhandled message types.

**Props:**
- `type: string` - The message part type
- `data: unknown` - The full data object

**Use case:** Debugging and handling future message types

**Visual:** Gray box with ❓ icon and JSON dump

---

## Usage Example

```svelte
<script lang="ts">
  import {
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
  } from "./message-parts";
</script>

{#each message.parts as part}
  {#if part.type === "text"}
    <TextPart text={part.text} />
  {:else if part.type.startsWith("tool-")}
    {@const toolName = part.type.replace("tool-", "")}
    {#if part.state === "input-available"}
      <ToolCallPart {toolName} input={part.input} state={part.state} />
    {:else if part.state === "output-available"}
      <ToolResultPart {toolName} output={part.output} />
    {/if}
  {:else if part.type === "data-reasoning"}
    {@const data = part.data as Record<string, unknown>}
    {@const status = typeof data.status === "string" ? data.status : "unknown"}
    <ReasoningPart {status} data={data} />
  {/if}
{/each}
```

## Design Principles

1. **Single Responsibility**: Each component handles one specific message type
2. **Prop Preparation**: Parent performs all string manipulation and type coercion
3. **Type Safety**: All components use TypeScript with proper interfaces
4. **Visual Consistency**: Color-coded boxes for easy scanning (yellow=tools, green=success, red=error, purple=reasoning)
5. **Svelte 5 Native**: Uses `$props()` rune for reactive props
6. **Debugging Support**: Unknown types always get a fallback display with full data dump

## File Structure

```
message-parts/
├── TextPart.svelte           # Regular text content
├── ToolCallPart.svelte       # Tool invocation (active)
├── ToolCalledPart.svelte     # Tool invocation (past)
├── ToolResultPart.svelte     # Tool output
├── ToolErrorPart.svelte      # Tool errors
├── ToolStatePart.svelte      # Other tool states
├── ToolNoStatePart.svelte    # Debug: tools without state
├── ReasoningPart.svelte      # Agent reasoning
├── UnknownPart.svelte        # Fallback for unknown types
├── index.ts                  # Barrel exports
└── README.md                 # This file
```

## Testing

All components have been validated with:
- ✅ TypeScript type checking (no errors)
- ✅ Svelte 5 autofixer (no issues)
- ✅ Linter checks (clean)

## Future Enhancements

Consider adding:
- Animation for state transitions
- Collapsible tool inputs/outputs for large data
- Copy-to-clipboard for JSON data
- Syntax highlighting for JSON output
- Loading spinners for streaming states

