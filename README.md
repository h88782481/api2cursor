# API 2 Cursor

让 Cursor 通过第三方中转站使用不同 LLM 协议的轻量转换网关。

Cursor 客户端只需要访问：

- `POST /v1/chat/completions`
- `GET /v1/models`

网关识别 Cursor 当前两种 Chat 工具方言：

- 标准 `function` 工具
- 带 grammar `format` 的 `custom` 工具

请求通过 [llm-rosetta](https://github.com/Oaklight/llm-rosetta) 的统一 IR 转换到四种上游协议：

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages
- Google Gemini GenAI

`/v1/responses` 与 `/v1/messages` 不作为客户端入口。

## 数据流

```text
Cursor Chat
  → Cursor 方言适配
  → llm-rosetta IR
  → 模型映射与请求策略
  → Chat / Responses / Messages / Gemini
  → llm-rosetta 流式或非流式回编
  → Cursor Chat
```

## 快速开始

### Docker 部署（推荐）

无需克隆仓库，只需要一个 `docker-compose.yml`：

```bash
# 下载 compose 文件
curl -O https://raw.githubusercontent.com/h88782481/api2cursor/main/docker-compose.yml
# 编辑 environment 中的中转站地址和密钥，然后启动
docker compose up -d
```

镜像发布在 `ghcr.io/h88782481/api2cursor`，推送 `v*` 标签时由 GitHub Actions 自动构建（支持 amd64 / arm64）。

想自行构建镜像的话，克隆仓库后把 compose 中的 `image:` 换成 `build: .` 即可。

### 直接运行

```bash
pip install -r requirements.txt
# 通过系统环境变量配置（也可以启动后在管理面板中配置中转站地址和密钥）
python main.py
```

服务启动后访问 `http://localhost:3029/admin` 进入管理面板。

### 发布新版本（维护者）

```bash
git tag v1.0.0
git push main v1.0.0   # 推送标签即触发自动构建发布
```

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
- **中转站接收格式** — `auto`（按上游模型名判断）/ `chat` / `messages` / `responses` / `gemini`
- **自定义地址/密钥** — 可选，覆盖全局设置，实现分流到不同中转站
- **思考等级** — 可选 `minimal` / `low` / `medium` / `high` / `xhigh` / `max`，按上游协议转换为对应的 reasoning/thinking 字段；具体可用等级取决于模型，不支持时上游会拒绝请求
- **Fast 模式** — 默认透传 Cursor 发送的 `service_tier`；映射中的开关可强制为 Chat / Responses 请求设置 `service_tier: "priority"`，需要上游支持且可能产生额外费用
- **自定义指令** — 按 Cursor 双方言（function / custom_grammar）分别配置，只改写 system 提示词
  - 目标：整段 system，或预设 XML 块（如 `tone_and_style`、`epistemic_rigor`）
  - 模式：前置 / 后置 / 覆盖（覆盖块时只替换块内文，保留标签）
  - 管理面板显示最近一次请求的注入状态；覆盖整段 system 时会进行风险确认
- **Body / Header 修改** — 对上游请求做字段级增删改（值为 `null` 删除）

**示例**：在 Cursor 中添加 `claude-sonnet-4-5-20250929`，映射到上游 `gpt-5.4`，中转站接收格式选 `responses`。请求会转换到 `/v1/responses`，响应再统一回编为 Cursor Chat。

### 在 Cursor 中配置

1. 打开 Cursor 设置 → Models
2. 添加自定义模型，名称填映射中配置的 Cursor 模型名
3. Override OpenAI Base URL 填 `http://你的服务器:3029`（需公网可达）
4. API Key 填 `ACCESS_API_KEY` 的值（未配置则随意填）

## 项目结构

```text
main.py                      # 启动入口 (uvicorn)
app/
├── __init__.py              # 应用组装、连接池与鉴权
├── api/
│   ├── chat.py              # /v1/chat/completions、/v1/models
│   └── admin.py             # 管理面板 + API
├── chat/
│   ├── gateway.py           # 单一 Chat 用例编排
│   ├── builder.py           # 上游请求构造
│   ├── cursor.py            # Cursor 双方言边界
│   ├── instructions.py      # system 提示词注入
│   ├── rosetta.py           # llm-rosetta 转换门面
│   ├── streaming.py         # 流式事件回编
│   └── exchange.py          # 请求上下文与错误
├── upstream/
│   ├── protocols.py         # URL、鉴权和 wire 规则
│   └── client.py            # HTTP 与 SSE
├── settings/
│   ├── schema.py            # 配置与环境变量模型
│   ├── repository.py        # 原子读写和缓存
│   └── resolver.py          # 模型映射解析
├── observability/
│   ├── request_log.py       # 对话日志
│   └── usage.py             # 用量统计
└── static/                  # 管理面板前端
```

## Custom grammar 工具

Responses 上游会保留 Cursor 原始 `custom` 工具和 grammar。Chat、Messages、Gemini 不支持原生 custom grammar 时，工具会降级为一个接收 `input` 字符串的函数，并在描述中保留格式提示。

回程时 custom 工具参数会恢复为 Cursor 需要的裸文本 `function.arguments`。

## 调试日志

- `off` — 关闭调试日志
- `simple` — 仅控制台调试日志
- `verbose` — 控制台 + 对话级文件日志，写入 `data/conversations/YYYY-MM-DD/{conversation_id}.json`，同一段多轮对话聚合到同一个文件，流式事件保留头尾各 12 条

## 许可证

[MIT](LICENSE)
