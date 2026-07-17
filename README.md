# MCP 工具市场

基于 Model Context Protocol 的标准化 AI 工具集合平台。包含 5 个 MCP Server、Agent 引擎和 Web 管理界面，并展示每个工具的类别、风险等级、权限边界和调用示例。

## 功能

| 工具 | 说明 |
|------|------|
| 📁 文件系统工具 | AI 读写文件、目录管理（安全沙箱隔离） |
| 🌤️ 天气查询工具 | 实时天气与天气预报（wttr.in） |
| 🧮 计算器工具 | 数学计算与单位换算（AST 安全求值） |
| 🗄️ 数据库查询工具 | 基于 SQLite 的结构化数据查询（SELECT-only） |
| 🧠 记忆与知识库 | Agent 长期记忆；文档切块、语义检索与来源引用（ChromaDB 持久化） |

Web 管理界面会把工具按市场卡片展示：连接状态、工具 schema、权限说明、安全边界和示例问题都在同一张卡片里，便于解释 Agent 工具治理。

> 安全边界：当前是本地演示系统，没有登录认证。`user_id` 只用于逻辑分区，不能当作生产级租户隔离；Docker 默认只把 Web 后端绑定到本机回环地址，MCP Server 仅在 Compose 内部网络可见。

## RAG 场景演示

连接“记忆系统”后，可以直接对 Agent 说：

1. `请将以下内容存入知识库，来源为员工手册：员工出差住宿标准为每晚500元，报销需在返程后10个工作日内提交。`
2. `根据知识库回答：出差住宿每晚最多报销多少？请标注来源。`

Agent 会先调用 `index_knowledge` 写入文档，再调用 `search_knowledge` 检索片段，并基于来源回答。
知识库是演示环境内共享语料，不按 `user_id` 分区；低于 `KNOWLEDGE_MIN_SCORE` 的片段不会返回，该阈值需要随 embedding 模型和真实评测校准。

运行知识库回归评测：

```bash
cd servers/memory-server
python evaluate_knowledge.py       # 离线管线回归，无需 API；不代表真实语义质量
python evaluate_knowledge.py live  # 使用当前环境变量配置的 Embedding API
```

## 快速开始

### 开发模式

```powershell
Copy-Item backend/.env.example backend/.env
# 编辑 backend/.env，填入 OpenAI 兼容 API 配置
python -m pip install -r backend/requirements.txt -r servers/memory-server/requirements.txt
python run.py          # 一键启动所有服务
```

浏览器打开 http://localhost:8000

### Docker 部署

```powershell
Copy-Item backend/.env.example backend/.env
# 编辑 backend/.env，填入 OpenAI 兼容 API 配置
docker compose up -d --build
```

`.env` 可省略以便先启动和测试 MCP 工具；AI 对话、语义记忆与知识库功能需要有效的 API Key 和支持的 Chat/Embedding 模型。

### 自检

```powershell
python smoke_test.py
python servers/memory-server/evaluate_knowledge.py
docker compose config --quiet
```

## 技术架构

```
MCP Server（FastMCP + SSE 传输）→ FastAPI 后端（MCP Client Manager + Agent 引擎）→ 原生 Web UI
```

每个 MCP Server 是独立服务，通过 SSE 传输实现 MCP 协议通信。后端运行 ReAct 循环，连接你的中转站 LLM 实现 AI 工具调用。

## 技术栈

- **MCP 协议**: mcp Python SDK（FastMCP）
- **后端**: Python / FastAPI / uvicorn
- **AI**: OpenAI 兼容 API
- **前端**: 原生 HTML/CSS/JS（无框架）
- **容器化**: Docker / Docker Compose

## 项目结构

```
mcp-tool-market/
├── servers/          # 5 个 MCP Server（每个独立端口）
│   ├── file-server/
│   ├── weather-server/
│   ├── calculator-server/
│   ├── db-server/
│   └── memory-server/  # 记忆系统（双阶段流水线 + ChromaDB）
├── backend/          # FastAPI 后端（端口 8000）
│   ├── app.py           # API 路由
│   ├── mcp_client.py    # MCP 客户端管理器
│   ├── agent_engine.py  # Agent ReAct 循环
│   └── static/          # Web 界面
├── docker-compose.yml
└── run.py
```

## License

MIT
