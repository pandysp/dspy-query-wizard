export interface AIQueryResponse {
  challenge: string;
  human_result: AIQueryResult;
  ai_result: AIQueryResult;
}

export interface AIQueryResult {
  prompt: string;
  llm_answer: string;
  score: number;
  chunks: AIQueryChunk[];
}

export interface AIQueryChunk {
  cosine_similarity: number;
  content: string;
  id: string;
}
