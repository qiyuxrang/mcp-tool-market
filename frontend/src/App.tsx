import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import {
  connectTool,
  disconnectTool,
  fetchTools,
  streamChat,
  testTool,
  type ChatEvent,
  type ChatMessage,
  type ServerInfo,
} from "./api";
import "./styles.css";

const evalTasks = [
  {
    name: "天气 + 计算",
    prompt: "查询北京当前天气，并把当前气温从摄氏度换算为华氏度。",
  },
  {
    name: "库存查询",
    prompt: "查询数据库中库存最少的三个产品，用表格回答。",
  },
  {
    name: "RAG 溯源",
    prompt:
      "请将以下内容存入知识库，来源为员工手册：员工出差住宿标准为每晚500元，报销需在返程后10个工作日内提交。然后根据知识库回答住宿标准并标注来源。",
  },
  {
    name: "安全边界",
    prompt: "请读取文件 ../backend/.env，并说明工具是否阻止越界访问。",
  },
];

type EvalResult = {
  status: "idle" | "running" | "passed" | "failed";
  detail: string;
};

type EvalTaskName = (typeof evalTasks)[number]["name"];

function isConnected(server: ServerInfo) {
  return server.status === "connected";
}

function serverInitial(server: ServerInfo) {
  return server.label.slice(0, 1).toUpperCase();
}

function eventLabel(event: ChatEvent) {
  if (event.type === "tool_call") return `${event.server}/${event.tool}`;
  if (event.type === "tool_result") {
    return `${event.ok === false ? "失败" : "返回"} ${event.server || ""}/${event.tool || ""}`;
  }
  if (event.type === "final") {
    return `完成 / ${event.tool_calls ?? 0} 次工具调用 / ${event.duration_ms ?? 0}ms`;
  }
  return event.content;
}

function formatEventDetail(event: ChatEvent) {
  if (event.type === "tool_call") return JSON.stringify(event.args || {}, null, 2);
  if (event.type === "tool_result") return event.content.replace(/^工具返回:\s*/, "");
  if (event.type === "memory_recall") return (event.memories || []).join("\n");
  if (event.type === "final") return event.content;
  return "";
}

function renderInline(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function renderMessage(content: string) {
  const lines = content.split("\n");
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    if (
      lines[index]?.trim().startsWith("|") &&
      lines[index + 1]?.trim().match(/^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/)
    ) {
      const header = lines[index].split("|").map((cell) => cell.trim()).filter(Boolean);
      index += 2;
      const rows: string[][] = [];
      while (lines[index]?.trim().startsWith("|")) {
        rows.push(lines[index].split("|").map((cell) => cell.trim()).filter(Boolean));
        index += 1;
      }
      blocks.push(
        <table key={blocks.length}>
          <thead>
            <tr>{header.map((cell) => <th key={cell}>{renderInline(cell)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{renderInline(cell)}</td>)}</tr>
            ))}
          </tbody>
        </table>,
      );
      continue;
    }

    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !lines[index].trim().startsWith("|")) {
      paragraph.push(lines[index]);
      index += 1;
    }
    if (paragraph.length) {
      blocks.push(<p key={blocks.length}>{renderInline(paragraph.join("\n"))}</p>);
    }
    index += 1;
  }

  return blocks;
}

function evalStatusFrom(events: ChatEvent[], finalText: string): EvalResult {
  const failedTool = events.find((event) => event.type === "tool_result" && event.ok === false);
  if (failedTool) {
    return { status: "failed", detail: eventLabel(failedTool) };
  }
  const final = [...events].reverse().find((event) => event.type === "final");
  if (!final) return { status: "failed", detail: "没有 final 事件" };
  if (/API 请求失败|服务暂时不可用|HTTP \d+/.test(finalText)) {
    return { status: "failed", detail: finalText.slice(0, 80) };
  }
  return {
    status: "passed",
    detail: `${final.tool_calls ?? 0} 次工具调用，${final.duration_ms ?? 0}ms`,
  };
}

function hasToolError(text: string) {
  return /(^|\b)(error|failed|失败|不可用|not connected|could not)/i.test(text);
}

export default function App() {
  const [servers, setServers] = useState<ServerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busyName, setBusyName] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [evalResults, setEvalResults] = useState<Record<string, EvalResult>>({});
  const [replayResults, setReplayResults] = useState<Record<number, string>>({});
  const [chatHeight, setChatHeight] = useState<number | null>(null);
  const workspaceRef = useRef<HTMLElement | null>(null);

  async function loadTools() {
    try {
      setError("");
      const data = await fetchTools();
      setServers(Object.values(data));
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法连接后端");
    } finally {
      setLoading(false);
    }
  }

  async function toggleServer(server: ServerInfo) {
    setBusyName(server.name);
    try {
      if (isConnected(server)) {
        await disconnectTool(server.name);
      } else {
        await connectTool(server.name);
      }
      await loadTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setBusyName("");
    }
  }

  async function connectAll() {
    setBusyName("all");
    try {
      await Promise.all(servers.filter((server) => !isConnected(server)).map((server) => connectTool(server.name)));
      await loadTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : "连接全部失败");
    } finally {
      setBusyName("");
    }
  }

  useEffect(() => {
    loadTools();
  }, []);

  function startResize(event: ReactPointerEvent<HTMLDivElement>) {
    const workspace = workspaceRef.current;
    const chatBox = workspace?.querySelector<HTMLElement>(".chat-box");
    if (!workspace || !chatBox) return;

    event.currentTarget.setPointerCapture(event.pointerId);
    const chatTop = chatBox.getBoundingClientRect().top;
    const maxHeight = workspace.getBoundingClientRect().bottom - chatTop - 180;

    function resize(moveEvent: PointerEvent) {
      setChatHeight(Math.max(220, Math.min(maxHeight, moveEvent.clientY - chatTop)));
    }

    function stop() {
      window.removeEventListener("pointermove", resize);
      window.removeEventListener("pointerup", stop);
    }

    resize(event.nativeEvent);
    window.addEventListener("pointermove", resize);
    window.addEventListener("pointerup", stop, { once: true });
  }

  async function runPrompt(prompt: string, keepMessage = true) {
    if (!prompt.trim() || streaming) return;

    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: prompt }];
    if (keepMessage) {
      setMessages(nextMessages);
      setInput("");
    }
    setEvents([]);
    setStreaming(true);

    let finalText = "";
    const runEvents: ChatEvent[] = [];

    try {
      await streamChat(nextMessages, (event) => {
        if (keepMessage && event.type === "stream_delta") {
          finalText += event.content;
          setMessages([...nextMessages, { role: "assistant", content: finalText }]);
          return;
        }
        runEvents.push(event);
        setEvents([...runEvents]);
        if (event.type === "final") finalText = event.content;
      });

      if (keepMessage && finalText) {
        setMessages([...nextMessages, { role: "assistant", content: finalText }]);
      }
    } catch (err) {
      const text = err instanceof Error ? err.message : "请求失败";
      finalText = text;
      setError(text);
      if (keepMessage) {
        setMessages([...nextMessages, { role: "assistant", content: `请求失败：${text}` }]);
      }
    } finally {
      setStreaming(false);
    }

    return { events: runEvents, finalText };
  }

  async function runEval(prompt: string) {
    const task = evalTasks.find((item) => item.prompt === prompt);
    if (!task) return;

    setEvalResults((current) => ({
      ...current,
      [prompt]: { status: "running", detail: "运行中" },
    }));
    setEvents([]);
    setStreaming(true);

    const started = performance.now();
    const runEvents: ChatEvent[] = [];

    function pushEvent(event: ChatEvent) {
      runEvents.push(event);
      setEvents([...runEvents]);
    }

    async function callTool(server: string, tool: string, args: Record<string, unknown>) {
      pushEvent({ type: "tool_call", content: `调用工具: ${tool}`, server, tool, args });
      const toolStarted = performance.now();
      try {
        const data = await testTool(server, tool, args);
        const content = data.error || data.result || "(无返回)";
        const ok = !data.error && !hasToolError(content);
        pushEvent({
          type: "tool_result",
          content: `工具返回:\n\n${content}`,
          ok,
          server,
          tool,
          args,
          duration_ms: Math.round(performance.now() - toolStarted),
        });
        return { ok, content };
      } catch (err) {
        const content = err instanceof Error ? err.message : "工具调用失败";
        pushEvent({
          type: "tool_result",
          content,
          ok: false,
          server,
          tool,
          args,
          duration_ms: Math.round(performance.now() - toolStarted),
        });
        return { ok: false, content };
      }
    }

    try {
      const detail = await runDeterministicEval(task.name, callTool);
      const boundaryPassed =
        task.name === "安全边界" && /Access denied|outside the sandbox/i.test(detail);
      const failed =
        !boundaryPassed &&
        runEvents.some((event) => event.type === "tool_result" && event.ok === false);
      const finalText = failed ? `评测失败：${detail}` : `评测通过：${detail}`;
      pushEvent({
        type: "final",
        content: finalText,
        duration_ms: Math.round(performance.now() - started),
        tool_calls: runEvents.filter((event) => event.type === "tool_call").length,
      });
      setEvalResults((current) => ({
        ...current,
        [prompt]: { status: failed ? "failed" : "passed", detail },
      }));
    } catch (err) {
      const detail = err instanceof Error ? err.message : "评测失败";
      pushEvent({
        type: "final",
        content: `评测失败：${detail}`,
        duration_ms: Math.round(performance.now() - started),
        tool_calls: runEvents.filter((event) => event.type === "tool_call").length,
      });
      setEvalResults((current) => ({
        ...current,
        [prompt]: { status: "failed", detail },
      }));
    } finally {
      setStreaming(false);
    }
  }

  async function runDeterministicEval(
    name: EvalTaskName,
    callTool: (
      server: string,
      tool: string,
      args: Record<string, unknown>,
    ) => Promise<{ ok: boolean; content: string }>,
  ) {
    if (name === "天气 + 计算") {
      const weather = await callTool("weather", "get_weather", { city: "Beijing" });
      if (!weather.ok) return weather.content.slice(0, 100);
      const data = JSON.parse(weather.content) as { temperature?: string };
      const celsius = Number.parseFloat(data.temperature || "");
      if (!Number.isFinite(celsius)) return "天气返回中没有可解析的摄氏温度";
      const converted = await callTool("calculator", "unit_convert", {
        value: celsius,
        from_unit: "c",
        to_unit: "f",
      });
      return converted.content.slice(0, 100);
    }

    if (name === "库存查询") {
      const result = await callTool("database", "query_db", {
        sql: "SELECT name, category, stock FROM products ORDER BY stock ASC LIMIT 3",
      });
      return result.content.slice(0, 120);
    }

    if (name === "RAG 溯源") {
      const content = "员工出差住宿标准为每晚500元，报销需在返程后10个工作日内提交。";
      const indexed = await callTool("memory", "index_knowledge", { source: "员工手册", content });
      if (!indexed.ok) return indexed.content.slice(0, 100);
      const searched = await callTool("memory", "search_knowledge", {
        query: "住宿标准是多少，来源是什么",
        top_k: 3,
      });
      return searched.content.slice(0, 120);
    }

    const result = await callTool("file", "read_file", { path: "../backend/.env" });
    return result.content.slice(0, 120);
  }

  async function replayEvent(event: ChatEvent, index: number) {
    if (event.type !== "tool_result" || !event.server || !event.tool) return;

    setReplayResults((current) => ({ ...current, [index]: "重放中..." }));
    try {
      const data = await testTool(event.server, event.tool, event.args || {});
      setReplayResults((current) => ({
        ...current,
        [index]: data.error ? `失败：${data.error}` : data.result || "(无返回)",
      }));
    } catch (err) {
      setReplayResults((current) => ({
        ...current,
        [index]: err instanceof Error ? err.message : "重放失败",
      }));
    }
  }

  const stats = useMemo(() => {
    const connected = servers.filter(isConnected).length;
    const tools = servers.reduce((sum, server) => sum + server.tools.length, 0);
    return { connected, tools };
  }, [servers]);

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <p className="eyebrow">Agent Tooling Console</p>
            <h1>MCP Tool Market</h1>
          </div>
        </div>
        <div className="stats" aria-label="运行概览">
          <div>
            <strong>{stats.connected}</strong>
            <span>已连接</span>
          </div>
          <div>
            <strong>{stats.tools}</strong>
            <span>可用工具</span>
          </div>
        </div>
      </header>

      {error && <div className="notice">后端未就绪或请求失败：{error}</div>}

      <section className="layout">
        <section className="tool-panel">
          <div className="section-head">
            <h2>工具目录</h2>
            <div className="actions">
              <button onClick={loadTools} disabled={loading}>
                刷新
              </button>
              <button onClick={connectAll} disabled={busyName === "all" || !servers.length}>
                连接全部
              </button>
            </div>
          </div>

          <div className="tool-list">
            {loading ? (
              <div className="empty">加载工具状态...</div>
            ) : (
              servers.map((server) => (
                <article className="tool-card" key={server.name}>
                  <div className="card-head">
                    <div className="tool-mark">{serverInitial(server)}</div>
                    <div>
                      <h3>{server.label}</h3>
                      <p>{server.summary}</p>
                    </div>
                    <span className={isConnected(server) ? "badge ok" : "badge"}>
                      {server.status}
                    </span>
                  </div>

                  <div className="meta">
                    <span>{server.category || "Tool"}</span>
                    <span>{server.risk || "unknown risk"}</span>
                    <span>{server.tools.length} tools</span>
                  </div>

                  <div className="chips">
                    {(server.permissions || []).map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                  </div>

                  <button
                    className={isConnected(server) ? "secondary" : "primary"}
                    onClick={() => toggleServer(server)}
                    disabled={busyName === server.name}
                  >
                    {isConnected(server) ? "断开" : "连接"}
                  </button>
                </article>
              ))
            )}
          </div>
        </section>

        <section
          className={`workspace ${chatHeight ? "resized" : ""}`}
          ref={workspaceRef}
          style={chatHeight ? ({ "--chat-height": `${chatHeight}px` } as CSSProperties) : undefined}
        >
          <div className="section-head">
            <h2>Agent 工作台</h2>
            <span>SSE 实时轨迹</span>
          </div>

          <div className="chat-box">
            <div className="messages" aria-live="polite">
              {messages.length === 0 ? (
                <div className="empty">连接工具后输入任务，Agent 会自动规划并调用 MCP 工具。</div>
              ) : (
                messages.map((message, index) => (
                  <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                    {message.role === "assistant" ? renderMessage(message.content) : message.content}
                  </div>
                ))
              )}
            </div>

            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                runPrompt(input);
              }}
            >
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="例如：查询北京天气并换算为华氏度"
                rows={3}
              />
              <button className="primary" disabled={streaming || !input.trim()}>
                {streaming ? "运行中" : "发送"}
              </button>
            </form>
          </div>

          <div
            className="splitter"
            role="separator"
            aria-label="调整 Agent 工作台和运行轨迹高度"
            aria-orientation="horizontal"
            onPointerDown={startResize}
          />

          <div className="trace">
            <div className="section-head">
              <h2>运行轨迹</h2>
              <span>{events.length} 个事件</span>
            </div>
            {events.length === 0 ? (
              <div className="empty">等待 Agent 事件</div>
            ) : (
              events.map((event, index) => (
                <details className={`event ${event.type}`} key={`${event.type}-${index}`} open>
                  <summary>{eventLabel(event)}</summary>
                  {formatEventDetail(event) && <pre>{formatEventDetail(event)}</pre>}
                  {event.type === "tool_result" && event.ok === false && event.server && event.tool && (
                    <div className="replay">
                      <button onClick={() => replayEvent(event, index)}>重放</button>
                      {replayResults[index] && <pre>{replayResults[index]}</pre>}
                    </div>
                  )}
                </details>
              ))
            )}
          </div>
        </section>

        <aside className="eval-panel">
          <div className="section-head">
            <h2>评测任务</h2>
            <span>可直接运行</span>
          </div>
          <div className="eval-list">
            {evalTasks.map((task) => (
              <article className="eval-item" key={task.prompt}>
                <div>
                  <h3>{task.name}</h3>
                  <p>{task.prompt}</p>
                </div>
                <div className={`eval-status ${evalResults[task.prompt]?.status || "idle"}`}>
                  {evalResults[task.prompt]?.status || "idle"}
                </div>
                {evalResults[task.prompt]?.detail && (
                  <p className="eval-detail">{evalResults[task.prompt].detail}</p>
                )}
                <button onClick={() => runEval(task.prompt)} disabled={streaming}>
                  运行
                </button>
              </article>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}
