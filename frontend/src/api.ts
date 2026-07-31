export type ToolSpec = {
  name: string;
  description?: string;
  inputSchema?: unknown;
};

export type ServerInfo = {
  name: string;
  label: string;
  status: string;
  category?: string;
  risk?: string;
  summary?: string;
  permissions?: string[];
  examples?: string[];
  tools: ToolSpec[];
};

export type ToolStatus = Record<string, ServerInfo>;

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatEvent =
  | {
      type: "thinking";
      content: string;
      trace_id?: string;
      round?: number;
    }
  | {
      type: "memory_recall";
      content: string;
      memories?: string[];
      trace_id?: string;
      duration_ms?: number;
    }
  | {
      type: "tool_call";
      content: string;
      server?: string;
      tool?: string;
      args?: Record<string, unknown>;
      trace_id?: string;
    }
  | {
      type: "tool_result";
      content: string;
      ok?: boolean;
      server?: string;
      tool?: string;
      args?: Record<string, unknown>;
      trace_id?: string;
      duration_ms?: number;
    }
  | {
      type: "final";
      content: string;
      trace_id?: string;
      duration_ms?: number;
      tool_calls?: number;
      tokens?: { prompt: number; completion: number; total: number };
    }
  | {
      type: "stream_delta";
      content: string;
      trace_id?: string;
    };

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function fetchTools() {
  return requestJson<ToolStatus>("/api/tools");
}

export function connectTool(name: string) {
  return requestJson<{ success: boolean; status: string }>("/api/tools/connect", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function disconnectTool(name: string) {
  return requestJson<{ success: boolean; error?: string }>("/api/tools/disconnect", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function testTool(
  server: string,
  tool: string,
  args: Record<string, unknown>,
  userId = "default",
) {
  return requestJson<{ result: string; error?: string }>("/api/tools/test", {
    method: "POST",
    body: JSON.stringify({
      server,
      tool,
      arguments: args,
      user_id: userId,
    }),
  });
}

export async function streamChat(
  messages: ChatMessage[],
  onEvent: (event: ChatEvent) => void,
  userId = "default",
) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, user_id: userId }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";

    for (const chunk of chunks) {
      const line = chunk.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      onEvent(JSON.parse(line.slice(6)) as ChatEvent);
    }
  }

  if (buffer.trim().startsWith("data: ")) {
    onEvent(JSON.parse(buffer.trim().slice(6)) as ChatEvent);
  }
}
