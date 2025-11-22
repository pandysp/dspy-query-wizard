import axios from "axios";
const API_URL = "http://127.0.0.1:8000";

// Use AXIOS
const api = axios.create({
  baseURL: API_URL,
});

export function getSystemMessages() {
  return api.get("/system_messages");
}

export function getInputPrompts() {
  return api.get("/input_prompts");
}
