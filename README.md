# API 2 Cursor

让 Cursor 通过第三方中转站使用任意 LLM 模型的协议转换代理。FastAPI + 统一中间表示（IR）架构。

## 它解决什么问题

Cursor 的 BYOK 有两个麻烦：

1. **请求格式由模型名决定**：Claude 风格模型名走 `/v1/chat/completions`（Chat Completions），GPT 风格模型名走 `/v1/responses`（Responses API），配 Anthropic Key 时走 `/v1/messages`。
2. **已知 bug**：Cursor 会把请求发到 `/v1/chat/completions`，但请求体实际是 Responses API 格式（无 `messages`，有 `input`/`instructions`），同时期望返回值仍是 Chat Completions 格式。

而中转站通常只支持其中一种协议。本项目在中间做协议转换：**你在映射里声明"Cursor 发送什么格式"和"中转站接收什么格式"，剩下的交给代理**——包括 bug 场景的自动识别。两侧格式相同时直接透传。

## 架构

任意组合不再写点对点转换，而是经过统一中间表示（IR）：每种协议只有一个编解码器，`Cursor 格式 → IR → 中转站格式`，回程反向；流式则是 `上游 SSE → IR 事件流 → 兼容过滤器 → Cursor SSE`。

```text
Cursor                          API 2 Cursor                        中转站
  │                                  │                                 │
  ├─ /v1/chat/completions ──┐        │        ┌─────────→ /v1/chat/completions
  │   (含 Responses 风格     │   ┌────┴────┐   │
  │    body 的 bug 场景)     ├──→│  解析 → IR │──┼─────────→ /v1/messages
  ├─ /v1/responses ─────────┤   │  IR → 构建 │  │
  │                         │   └────┬────┘   ├─────────→ /v1/responses
  └─ /v1/messages ──────────┘        │        └─────────→ Gemini generateContent
                                     │
                          两侧格式相同时直接透传
```

- 入口协议：`chat` / `responses` / `messages`（响应格式始终匹配请求到达的入口）
- 上游协议：`chat` / `messages` / `responses` / `gemini`
- BYOK bug 处理：`chat` 入口收到 Responses 风格请求体时自动按 Responses 解析，返回时仍编码为 Chat Completions

## 快速开始

### Docker 部署（推荐）

```bash
# 编辑 docker-compose.yml 中的 environment 配置，填入中转站地址和密钥
docker compose up -d
```

### 直接运行

```bash
pip install -r requirements.txt
# 通过系统环境变量配置（也可以启动后在管理面板中配置中转站地址和密钥）
python main.py
```

服务启动后访问 `http://localhost:3029/admin` 进入管理面板。

## 配置

### 环境变量

环境变量统一在 `docker-compose.yml` 的 `environment` 中配置（直接运行时通过系统环境变量设置）：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `PROXY_TARGET_URL` | 上游中转站地址 | `https://api.anthropic.com` |
| `PROXY_API_KEY` | 上游 API 密钥 | |
| `PROXY_PORT` | 服务监听端口（修改时需同步修改 compose 的 ports 映射） | `3029` |
| `API_TIMEOUT` | 请求超时（秒） | `300` |
| `ACCESS_API_KEY` | 访问鉴权密钥，留空不启用 | |
| `DEBUG_MODE` | 调试模式：`off` / `simple` / `verbose` | `off` |

中转站地址与密钥也可以在管理面板全局设置中配置，面板配置优先于环境变量。

### 模型映射

在管理面板 (`/admin`) 中配置：

- **Cursor 模型名** — 在 Cursor 自定义模型中填入的名称
- **上游模型名** — 发送到中转站的实际模型名
- **Cursor 发送格式** — `auto`（跟随请求实际到达的入口，推荐）/ `chat` / `responses` / `messages`
- **中转站接收格式** — `auto`（按上游模型名判断）/ `chat` / `messages` / `responses` / `gemini`
- **自定义地址/密钥** — 可选，覆盖全局设置，实现分流到不同中转站
- **自定义指令** — 注入到上游请求的系统提示词（可选前置/后置）
- **Body / Header 修改** — 对上游请求做字段级增删改（值为 `null` 删除）

**示例**：在 Cursor 中添加 `claude-sonnet-4-5-20250929`，映射到上游 `gpt-5.4`，中转站接收格式选 `responses`。Cursor 发来的请求（无论是标准 CC 还是 Responses 风格 body）都会被转换为 `/v1/responses` 请求，响应再转回 Cursor 期望的格式。

> **提示**：使用 Claude 风格的模型名可以让 Cursor 显示思考过程（thinking）。旧版 `backend` 字段的 `data/settings.json` 会在启动时自动迁移。

### 在 Cursor 中配置

1. 打开 Cursor 设置 → Models
2. 添加自定义模型，名称填映射中配置的 Cursor 模型名
3. Override OpenAI Base URL 填 `http://你的服务器:3029`（需公网可达）
4. API Key 填 `ACCESS_API_KEY` 的值（未配置则随意填）

## 项目结构

```text
main.py                      # 启动入口 (uvicorn)
app/
├── __init__.py              # 应用工厂 + lifespan(httpx 连接池) + 鉴权中间件
├── config.py                # 环境变量 (pydantic-settings)
├── store.py                 # data/settings.json 持久化 + 旧格式迁移 + 模型映射解析
├── api/
│   ├── entry.py             # /v1/chat/completions、/v1/responses、/v1/messages、/v1/models
│   └── admin.py             # 管理面板 + API
├── core/
│   ├── ir.py                # 统一中间表示：请求/响应/流式事件
│   ├── routing.py           # 模型映射 → 路由决策
│   ├── pipeline.py          # 编排：转换与透传两条路径（流式 + 非流式）
│   └── upstream.py          # httpx 转发、SSE 解析
├── protocols/               # 每种协议一个编解码器
│   ├── chat_completions.py  # CC 双向
│   ├── responses_api.py     # Responses 双向（含 prompt_cache_key）
│   ├── anthropic.py         # Messages 双向（含 cache_control、max_tokens 兜底）
│   └── gemini.py            # Gemini 仅上游方向
├── compat/
│   ├── detect.py            # Responses 风格请求体检测（BYOK bug）
│   ├── tools.py             # 工具定义规范化 + 参数修复
│   └── thinking.py          # <think> 标签提取 + thinking 缓存 + messages 透传注入
├── services/
│   ├── request_log.py       # 三档调试日志（verbose 写对话级文件）
│   └── usage.py             # 用量统计
└── static/                  # 管理面板前端
```

## 兼容性处理

- Responses 风格请求体误入 `/v1/chat/completions` 自动识别（`input`/`instructions`/`reasoning`/`text` 等标记），并剥离 Cursor 混入的 CC 专属字段（`stream_options`/`metadata`，原样转发会被 `/v1/responses` 上游拒绝）
- Cursor 扁平工具定义 → 标准格式；Anthropic 风格 `tool_use`/`tool_result` 块混入 CC 消息的转换
- `reasoningContent` → `reasoning_content`；`<think>` 标签 → 思考内容（流式跨块解析）
- 旧版 `function_call` → `tool_calls`；流式 tool_calls 空白 id/name 清理与元数据补全
- StrReplace 智能引号容错修复、`file_path` → `path`
- 多轮对话 thinking 缓存回注（Cursor 不回传思考内容，推理模型缺失历史 thinking 时降质）
- Anthropic 上游自动 `cache_control`（顶层自动提示缓存）；Responses 上游自动 `prompt_cache_key`
- Gemini 上游走 v1beta 端点（v1 不支持函数调用）；Gemini 3 强制回传的 `thoughtSignature` 按 tool_call_id 缓存并在多轮工具调用时重新附加，未命中时用官方哨兵值兜底
- messages 透传时非标准 `reasoning_content` → 标准 thinking block（含流式 index 偏移）

## 调试日志

- `off` — 关闭调试日志
- `simple` — 仅控制台调试日志
- `verbose` — 控制台 + 对话级文件日志，写入 `data/conversations/YYYY-MM-DD/{conversation_id}.json`，同一段多轮对话聚合到同一个文件，流式事件保留头尾各 12 条

## 许可证

[MIT](LICENSE)
