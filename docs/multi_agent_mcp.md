# Multi-agent 协作与 MCP

现有规划、研究、分析、审阅、审批和导出流程保持兼容。`POST /runs` 可以通过可选的 `collaboration` 配置，在规划完成后启动自定义专家团队。前端界面无需修改；连接管理与协作配置通过 API、CLI 或 Python Runner 使用。

## 自定义专家团队

```json
{
  "task": "根据运营数据库分析服务可靠性，提出改进建议。",
  "task_id": "database-review",
  "collaboration": {
    "mode": "sequential",
    "max_tool_calls": 4,
    "agents": [
      {"name": "database", "role": "查询相关运营数据，说明数据支持的事实和局限。", "mcp_servers": ["operations"]},
      {"name": "reviewer", "role": "检查上一位专家的结论，列出风险、分歧和待验证假设。"}
    ]
  }
}
```

- `mode=parallel`：通过 LangGraph `Send` 并发分派，每位专家独立处理同一个任务和计划。
- `mode=sequential`：按配置顺序执行，后一位专家收到之前所有专家的结构化结果。
- 每次最多 8 位专家，名称必须唯一。`max_tool_calls` 限制每位专家的实际工具调用次数（0–16），不计工具发现；模型调用最多为该值加 1。
- 专家只能使用自己 `mcp_servers` 中声明的连接，以及服务器 `allowed_tools` 放行的工具。无需 MCP 的专家可以省略该字段。
- 专家使用当前配置的 LLM Provider。工具输出和专家结论作为数据处理，不授予新的工具权限。
- MCP 工具成功返回的原始数据转换为带 `mcp://` 来源定位的研究资料，经现有解析、评分、证据引用、分析和治理流程处理。专家摘要供分析员参考，不直接当作已验证证据。
- 专家失败记录为 `status=error`，已有资料继续供研究使用；进入分析后会触发 `specialist_execution_failed` 人工审阅策略。

响应增加 `collaboration_results`，已有响应字段保留。专家和 MCP 调用会写入现有的 `execution_trace`、`tool_call_history` 和 `model_call_history`；`run_artifact.json` 与 `state_snapshot.json` 保存协作配置与结果。SQLite 检查点支持审批暂停后的重启恢复。

CLI 将 `collaboration` 对象（不含外层 `task`）保存为 JSON 后执行：

```bash
python -m app.main --task "分析运营数据" --collaboration-file team.json
```

Python 调用：

```python
runner.start(task="分析运营数据", collaboration={
    "agents": [{"name": "expert", "role": "审阅架构和运维风险"}]
})
```

## 启用 MCP

依赖已加入 `pyproject.toml` 与 `requirements.lock`。这里使用[官方 Python SDK 的 v1 维护分支](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)，依赖上限为 `<2`，避免协议客户端 API 升级改变运行行为。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `MCP_ENABLED` | `false` | 开启连接管理和协议调用 |
| `MCP_ALLOW_STDIO` | `false` | 允许启动本地 MCP 子进程 |
| `MCP_ALLOWED_COMMANDS` | 空 | 允许的 stdio 可执行文件，逗号分隔，精确匹配 |
| `MCP_ALLOWED_ENV_VARS` | 空 | 可用于连接的环境变量名称，逗号分隔 |
| `MCP_TIMEOUT` | `30` | 单次操作的总超时（秒，含启动、初始化、分页和调用） |
| `MCP_MAX_OUTPUT_CHARS` | `32000` | 工具和资源内容的长度预算；超出时返回 `truncated=true` |
| `MCP_MAX_TOOLS` | `64` | 工具/资源发现的最大条目数 |

连接保存到 `OUTPUT_ROOT/mcp.sqlite`，API 与 RabbitMQ workers 必须共享该目录及相同 MCP 配置。环境变量中的秘密值不会写入连接记录；协议调用时才解析引用，错误响应也不暴露底层诊断。每次操作使用独立 SDK 会话并关闭传输和子进程。

沿用项目现有部署边界：API 位于可信的认证网关后，连接注册表属于整个服务实例，尚未提供用户/租户隔离。连接管理权限应由网关控制。stdio 命令参数属于受信任的服务器配置；不应在 `args` 或 URL 中放置秘密值，应通过受允许运维环境变量传递。授权工具的实际数据库权限由 MCP 服务端及数据库账号决定。

## 连接远程 MCP 服务

设置 `MCP_ENABLED=true`。无认证连接示例：

```bash
curl -X POST http://127.0.0.1:8000/mcp/servers \
  -H 'Content-Type: application/json' \
  -d '{"name":"operations","transport":"streamable_http","url":"http://127.0.0.1:9000/mcp","allowed_tools":["query"]}'
```

已有 SSE 服务可以使用 `transport=sse` 和其 `/sse` 端点。URL 必须使用 HTTP(S)，不接受用户名、密码、查询参数或 fragment。

需要 Bearer 认证时，在 API 和 worker 环境中设置例如 `OPERATIONS_MCP_AUTH='Bearer ...'`，并把 `OPERATIONS_MCP_AUTH` 加入 `MCP_ALLOWED_ENV_VARS`；连接 JSON 中使用：

```json
{"headers_env": {"Authorization": "OPERATIONS_MCP_AUTH"}}
```

`headers_env` 的值是变量名，变量值应包含完整认证头。服务端工具中回显的这些变量值也会在工具/资源响应中隐藏。

## 连接 SQLite 数据库

仓库提供 `examples/mcp_sqlite_server.py`，暴露只读 `query(sql)` 工具及 `database://schema` 资源。它使用 SQLite 只读连接和 authorizer，限制返回 100 行并约束查询执行量，拒绝写入和 ATTACH。

在 API 与 worker 的环境中配置以下值，将路径替换为本机实际路径：

```env
MCP_ENABLED=true
MCP_ALLOW_STDIO=true
MCP_ALLOWED_COMMANDS=/absolute/project/.venv/bin/python
MCP_ALLOWED_ENV_VARS=OPERATIONS_DATABASE_PATH
OPERATIONS_DATABASE_PATH=/absolute/data/operations.sqlite
```

注册连接：

```json
{
  "name": "operations",
  "transport": "stdio",
  "command": "/absolute/project/.venv/bin/python",
  "args": ["/absolute/project/examples/mcp_sqlite_server.py"],
  "env": {"MCP_SQLITE_PATH": "OPERATIONS_DATABASE_PATH"},
  "allowed_tools": ["query"]
}
```

POST 到 `/mcp/servers`，随后调用：

```bash
curl -X POST http://127.0.0.1:8000/mcp/servers/operations/tools/query/call \
  -H 'Content-Type: application/json' \
  -d '{"arguments":{"sql":"SELECT name FROM sqlite_master WHERE type = '\''table'\''"}}'
```

PostgreSQL、MySQL 或其他数据服务通过相应的 MCP 服务端接入相同接口；数据库连接参数由该 MCP 服务端处理。

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET / POST | `/mcp/servers` | 列出 / 创建连接（创建返回 201） |
| GET / PUT / DELETE | `/mcp/servers/{name}` | 查询 / 完整替换 / 删除连接 |
| POST | `/mcp/servers/{name}/test` | 初始化连接并检查工具发现 |
| GET | `/mcp/servers/{name}/tools` | 工具及输入 schema；`allowed` 表示是否允许调用 |
| POST | `/mcp/servers/{name}/tools/{tool_name}/call` | 调用工具，body 为 `{"arguments": {...}}` |
| GET | `/mcp/servers/{name}/resources` | 列出资源 |
| POST | `/mcp/servers/{name}/resources/read` | 读取资源，body 为 `{"uri":"..."}` |

`allowed_tools=[]` 允许发现工具，但拒绝调用；`["*"]` 显式放行全部工具。连接可设置 `enabled=false` 暂停调用。PUT 中的 `name` 必须与路径一致。

不存在的连接返回 404、重复创建或禁用连接返回 409、策略拒绝返回 403、参数问题返回 422、协议/网络/超时错误返回 502、未开启 MCP 返回 503。工具执行错误保留 MCP 的 `isError=true` 结果，不包装成网络错误。调用不会自动重试；任务进程故障后的重放仍可能再次执行尚未完成检查点提交的工具，存在副作用的工具须由 MCP 服务端提供幂等机制。
