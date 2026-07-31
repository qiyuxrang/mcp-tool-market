# MCP Tool Market

一个面向企业 Agent 应用的工具治理与调用工作台。项目把文件、天气、计算、数据库、知识库等能力封装为独立 MCP Server，由后端 Agent 统一发现、编排和调用，并在 React 控制台中展示连接状态、权限边界、流式回答和完整工具轨迹。

## 商业价值

企业落地 Agent 时，真正的难点通常不是“让模型回答问题”，而是让模型安全、可观测地调用业务工具：

- **工具标准化**：把不同系统能力统一成 MCP 工具，降低 Agent 接入成本。
- **权限可解释**：每个工具展示风险等级、权限边界和调用范围，便于治理与审计。
- **调用可追踪**：前端实时展示 thinking、tool_call、tool_result、final，方便排查模型为什么这么做。
- **安全边界**：文件沙箱、数据库 SELECT-only、表达式白名单、用户记忆逻辑分区，展示基本防护意识。
- **评测闭环**：内置天气计算、库存查询、RAG 溯源和安全边界评测，便于演示稳定能力。

这个项目适合包装为：**企业内部 Agent 工具市场 / MCP 工具治理控制台 / AI 应用开发平台原型**。

## 推荐截图

面试和简历材料建议准备 4 张图：

| 文件名建议 | 截图内容 | 用途 |
|---|---|---|
| `01-dashboard.png` | 三栏 React 控制台，左侧 5 个工具全部 connected，中间 Agent 工作台，右侧评测任务 | 展示“工具市场 + 控制台”整体形态 |
| `02-weather-trace.png` | 运行“天气 + 计算”后，中间回答区和运行轨迹同时可见 | 展示 Agent 自动调用多个 MCP 工具 |
| `03-rag-source.png` | RAG 溯源任务通过，回答中包含员工手册来源 | 展示知识库检索和来源约束 |
| `04-security-boundary.png` | 安全边界任务通过，读取 `../backend/.env` 被拒绝 | 展示权限边界和安全意识 |

截图重点不是 UI 多漂亮，而是让面试官一眼看到：**工具连接状态、工具调用轨迹、最终回答、评测结果**。

## 核心能力

| 模块 | 说明 | 风险控制 |
|---|---|---|
| 文件系统 | 在沙箱目录中读写文件、列目录 | 阻止目录穿越、限制文件大小 |
| 天气查询 | 通过 wttr.in 查询实时天气和预报 | 只读网络请求 |
| 计算器 | 数学计算和单位换算 | AST 白名单，无 `eval` |
| 数据库查询 | 查询示例 SQLite 库存数据 | SELECT-only，限制返回行数 |
| 记忆与知识库 | 长期记忆、文档索引、语义检索、来源引用 | user_id 逻辑分区，检索片段带来源 |

## 演示场景

1. **天气 + 计算**
   用户询问“查询北京当前天气，并把气温换算为华氏度”，Agent 会先调用天气工具，再调用计算器工具。

2. **库存查询**
   用户询问“库存最少的三个产品”，Agent 调用数据库工具并用表格回答。

3. **RAG 溯源**
   用户写入员工手册内容后，Agent 调用知识库检索并基于来源回答制度问题。

4. **安全边界**
   用户尝试读取 `../backend/.env`，文件工具会拒绝沙箱外访问。

## 技术架构

```text
React 控制台
  -> FastAPI SSE API
  -> ReAct Agent Engine
  -> MCP Client Manager
  -> 5 个独立 MCP Server
  -> 文件 / Web API / AST 计算 / SQLite / ChromaDB
```

## 技术栈

- **前端**：React + TypeScript + Vite
- **后端**：Python + FastAPI + SSE
- **Agent**：OpenAI 兼容 Chat Completions，支持流式输出和 function calling
- **MCP**：mcp Python SDK + FastMCP + SSE transport
- **RAG/记忆**：ChromaDB + OpenAI 兼容 Embedding
- **部署**：Docker Compose

## 快速开始

```powershell
Copy-Item backend/.env.example backend/.env
# 编辑 backend/.env，填入 OpenAI 兼容 API 配置

python -m pip install -r backend/requirements.txt -r servers/memory-server/requirements.txt
cd frontend
npm install
npm run build
cd ..
python run.py
```

浏览器打开：

- React 开发服务：`http://127.0.0.1:5174`
- 后端托管页面：`http://127.0.0.1:8000`

## Docker 部署

```powershell
Copy-Item backend/.env.example backend/.env
# 编辑 backend/.env
docker compose up -d --build
```

Docker 镜像会构建 React 前端并由 FastAPI 托管。

## 自检

```powershell
python smoke_test.py
cd frontend
npm run build
cd ..
docker compose config --quiet
```

## 项目结构

```text
mcp-tool-market/
├── frontend/             # React + TypeScript 控制台
├── backend/              # FastAPI API、Agent 引擎、MCP Client Manager
├── servers/
│   ├── file-server/      # 文件沙箱 MCP Server
│   ├── weather-server/   # 天气 MCP Server
│   ├── calculator-server/# AST 计算 MCP Server
│   ├── db-server/        # SQLite 查询 MCP Server
│   └── memory-server/    # 记忆与知识库 MCP Server
├── smoke_test.py
├── docker-compose.yml
└── run.py
```

## 当前边界

这是本地演示和面试项目，不是生产 SaaS：

- 暂无登录、租户鉴权和 RBAC。
- `user_id` 只做逻辑分区，不能当生产级隔离。
- 天气工具依赖公开 wttr.in，稳定性受外部服务影响。
- 知识库阈值需要随真实 embedding 模型和业务语料继续校准。

## License

MIT
