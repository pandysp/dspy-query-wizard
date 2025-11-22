import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { AIQueryResponse } from "./types";

export function cn(...classes: ClassValue[]) {
  return twMerge(clsx(...classes));
}
