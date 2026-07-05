# MCP 工具市场

基于 Model Context Protocol 的标准化 AI 工具集合平台。包含 4 个 MCP Server、Agent 引擎和 Web 管理界面。

## 功能

| 工具 | 说明 |
|------|------|
| 📁 文件系统工具 | AI 读写文件、目录管理（安全沙箱隔离） |
| 🌤️ 天气查询工具 | 实时天气与天气预报（wttr.in） |
| 🧮 计算器工具 | 数学计算与单位换算（AST 安全求值） |
| 🗄️ 数据库查询工具 | 基于 SQLite 的结构化数据查询（SELECT-only） |
| 🧠 记忆系统 | Agent 长期记忆（Mem0 式双阶段流水线，ChromaDB 持久化） |

## 快速开始

### 开发模式

```bash
cd backend
cp .env.example .env   # 编辑填入你的中转站 API 配置
pip install -r requirements.txt
cd ..
python run.py          # 一键启动所有服务
```

浏览器打开 http://localhost:8000

### Docker 部署

```bash
cp backend/.env.example backend/.env
docker-compose up -d
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
