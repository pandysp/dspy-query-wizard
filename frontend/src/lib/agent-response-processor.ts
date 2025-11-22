export type ToolCall = {
  id: string;
  name: string;
  input?: Record<string, unknown>;
  output?: unknown;
  status: 'pending' | 'complete' | 'error';
  errorText?: string;
};

export type AgentPhase = {
  type: 'research' | 'reasoning' | 'answer';
  id: string;
  startTime: number;
  endTime?: number;

  // Research phase
  toolCalls?: ToolCall[];

  // Reasoning phase
  thoughts?: {
    text: string;
    isStreaming: boolean;
  };

  // Answer phase
  answer?: {
    text: string;
    isStreaming: boolean;
  };
};

export type StreamEvent = {
  type: string;
  id?: string;
  toolCallId?: string;
  toolName?: string;
  input?: Record<string, unknown>;
  output?: unknown;
  delta?: string;
  text?: string;
  errorText?: string;
};

export class AgentResponseProcessor {
  private phases: AgentPhase[] = [];
  private currentPhase: AgentPhase | null = null;
  private currentToolCall: ToolCall | null = null;
  private currentReasoningId: string | null = null;
  private currentTextId: string | null = null;

  getPhases(): AgentPhase[] {
    // Return copy with current phase appended if active
    const result = [...this.phases];
    if (this.currentPhase) {
      result.push(this.currentPhase);
    }
    return result;
  }

  processEvent(event: StreamEvent): void {
    switch (event.type) {
      // Tool events
      case 'tool-input-start':
        this.handleToolInputStart(event);
        break;
      case 'tool-input-available':
        this.handleToolInputAvailable(event);
        break;
      case 'tool-output-available':
        this.handleToolOutputAvailable(event);
        break;

      // Reasoning events
      case 'reasoning-start':
        this.handleReasoningStart(event);
        break;
      case 'reasoning-delta':
        this.handleReasoningDelta(event);
        break;
      case 'reasoning-end':
        this.handleReasoningEnd(event);
        break;

      // Text events
      case 'text-start':
        this.handleTextStart(event);
        break;
      case 'text-delta':
        this.handleTextDelta(event);
        break;
      case 'text-end':
        this.handleTextEnd(event);
        break;

      case 'finish':
        this.completeCurrentPhase();
        break;
    }
  }

  private handleToolInputStart(event: StreamEvent): void {
    // Start research phase if not in one
    if (!this.currentPhase || this.currentPhase.type !== 'research') {
      this.completeCurrentPhase();
      this.startPhase('research');
    }

    // Create new tool call
    this.currentToolCall = {
      id: event.toolCallId || crypto.randomUUID(),
      name: event.toolName || 'unknown',
      status: 'pending',
    };
  }

  private handleToolInputAvailable(event: StreamEvent): void {
    if (this.currentToolCall && event.input) {
      this.currentToolCall.input = event.input;
    }
  }

  private handleToolOutputAvailable(event: StreamEvent): void {
    if (this.currentToolCall) {
      this.currentToolCall.output = event.output;
      this.currentToolCall.status = 'complete';

      // Add to current research phase
      if (this.currentPhase?.type === 'research') {
        if (!this.currentPhase.toolCalls) {
          this.currentPhase.toolCalls = [];
        }
        this.currentPhase.toolCalls.push(this.currentToolCall);
      }

      this.currentToolCall = null;
    }
  }

  private handleReasoningStart(event: StreamEvent): void {
    // Complete current phase and start reasoning phase
    this.completeCurrentPhase();
    this.startPhase('reasoning');
    this.currentReasoningId = event.id || crypto.randomUUID();

    if (this.currentPhase) {
      this.currentPhase.thoughts = {
        text: '',
        isStreaming: true,
      };
    }
  }

  private handleReasoningDelta(event: StreamEvent): void {
    if (this.currentPhase?.type === 'reasoning' && event.delta) {
      if (!this.currentPhase.thoughts) {
        this.currentPhase.thoughts = { text: '', isStreaming: true };
      }
      this.currentPhase.thoughts.text += event.delta;
    }
  }

  private handleReasoningEnd(event: StreamEvent): void {
    if (this.currentPhase?.type === 'reasoning' && this.currentPhase.thoughts) {
      this.currentPhase.thoughts.isStreaming = false;
    }
    this.currentReasoningId = null;
    this.completeCurrentPhase();
  }

  private handleTextStart(event: StreamEvent): void {
    // If we're in research phase, complete it and start answer phase
    if (this.currentPhase?.type === 'research') {
      this.completeCurrentPhase();
    }

    // Start answer phase if not already in one
    if (!this.currentPhase || this.currentPhase.type !== 'answer') {
      this.startPhase('answer');
    }

    this.currentTextId = event.id || crypto.randomUUID();

    if (this.currentPhase) {
      this.currentPhase.answer = {
        text: '',
        isStreaming: true,
      };
    }
  }

  private handleTextDelta(event: StreamEvent): void {
    if (this.currentPhase?.type === 'answer' && event.delta) {
      if (!this.currentPhase.answer) {
        this.currentPhase.answer = { text: '', isStreaming: true };
      }
      this.currentPhase.answer.text += event.delta;
    }
  }

  private handleTextEnd(event: StreamEvent): void {
    if (this.currentPhase?.type === 'answer' && this.currentPhase.answer) {
      this.currentPhase.answer.isStreaming = false;
    }
    this.currentTextId = null;
  }

  private startPhase(type: AgentPhase['type']): void {
    this.currentPhase = {
      type,
      id: crypto.randomUUID(),
      startTime: Date.now(),
    };
  }

  private completeCurrentPhase(): void {
    if (this.currentPhase) {
      this.currentPhase.endTime = Date.now();
      this.phases.push(this.currentPhase);
      this.currentPhase = null;
    }
  }

  reset(): void {
    this.phases = [];
    this.currentPhase = null;
    this.currentToolCall = null;
    this.currentReasoningId = null;
    this.currentTextId = null;
  }
}

export function formatDuration(startTime: number, endTime?: number): string {
  if (!endTime) return '';
  const seconds = (endTime - startTime) / 1000;
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  return `${seconds.toFixed(1)}s`;
}

export function parseWikipediaResult(item: string | unknown): { title: string; snippet: string; fullContent: string } {
  if (typeof item !== 'string') {
    return { title: 'Unknown', snippet: JSON.stringify(item), fullContent: JSON.stringify(item) };
  }

  const separatorIndex = item.indexOf(' | ');
  if (separatorIndex === -1) {
    return { title: 'Unknown', snippet: item, fullContent: item };
  }

  const title = item.substring(0, separatorIndex);
  const content = item.substring(separatorIndex + 3);

  // Truncate snippet intelligently (at sentence boundary)
  const snippetLimit = 150;
  let snippet = content;
  if (content.length > snippetLimit) {
    const truncated = content.substring(0, snippetLimit);
    const lastPeriod = truncated.lastIndexOf('. ');
    snippet = lastPeriod > 50
      ? truncated.substring(0, lastPeriod + 1)
      : truncated + '...';
  }

  return { title, snippet, fullContent: content };
}
