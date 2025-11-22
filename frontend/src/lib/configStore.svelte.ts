import { getInputPrompts, getSystemMessages } from "./api";

export const sytemMessages = $state({
  default: "",
  optimized: "",
});

export const inputPrompts = $state<{
  prompts: {
    id: string;
    prompt: string;
  }[];
}>({ prompts: [] });

export const selectedInputPrompt = $state<{
  id: string;
}>({ id: "" });

export async function loadSystemMessages() {
  const response = await getSystemMessages();
  sytemMessages.default = response.data.default;
  sytemMessages.optimized = response.data.optimized;
}

export async function loadInputPrompts() {
  const response = await getInputPrompts();
  selectedInputPrompt.id = response.data[0].id;
  inputPrompts.prompts = response.data;
}
