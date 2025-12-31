# LangChain 学习笔记

## Agent (代理)

### 核心概念

Agent 是 LangChain 中的智能体，能够使用工具(Tools)来执行任务并响应用户请求。

---

### 1. 静态模型 (Static Model)

最基础的 Agent 创建方式，模型在创建时配置一次，整个执行过程保持不变。

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

model = ChatOpenAI(
    model="glm-4",
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)

agent = create_agent(model, tools=[])

# 调用
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
print(result["messages"][-1].content)
```

---

### 2. 动态模型 (Dynamic Model)

通过 `@wrap_model_call` 中间件，根据对话复杂度动态切换模型。

**场景**：简单对话用快速模型，复杂对话用高级模型。

```python
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 定义两个模型
basic_model = ChatOpenAI(model="glm-4-flash", api_key=api_key)
advanced_model = ChatOpenAI(model="glm-4", api_key=api_key)

# 定义动态模型选择函数
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话轮数选择模型"""
    message_count = len(request.state["messages"])  # 获取对话轮数
    if message_count > 10:
        model = advanced_model  # 超过10轮使用高级模型
    else:
        model = basic_model     # 前10轮使用快速模型
    return handler(request.override(model=model))

# 创建 Agent（注意：源代码中注释掉了，这里展示完整用法）
agent = create_agent(
    model=basic_model,
    tools=[],
    middleware=[dynamic_model_selection]
)

# 使用
config = {"configurable": {"thread_id": "chat-001"}}
for i in range(12):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"Message {i+1}"}]},
        config=config
    )
    # 前10轮使用 basic_model，第11轮开始使用 advanced_model
```

**关键点**：
- `request.state["messages"]` - 访问对话历史
- `request.override(model=model)` - 动态替换模型
- 对话历史会被完整传递，不会丢失上下文

---

### 3. 工具定义 (Tools)

工具让 Agent 能够执行外部操作。

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

**工具可以做什么？**

| 功能 | 实现方式 |
|------|----------|
| 查询数据库 | 连接 MySQL/PostgreSQL/MongoDB |
| 调用内部 API | 发送 HTTP 请求 |
| 搜索互联网 | 集成 Google Search、Bing API |
| 读写文件 | 本地文档操作 |

---

### 4. 工具错误处理

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """处理工具执行错误"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model,
    tools=[...],
    middleware=[handle_tool_errors]
)
```

---

### 5. 动态系统提示 (@dynamic_prompt)

根据用户角色动态生成不同的系统提示。

```python
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model=model,
    middleware=[user_role_prompt],
    context_schema=Context
)

# 使用：根据 context 动态切换提示
result = agent.invoke(
    {"messages": [HumanMessage("Explain machine learning")]},
    context={"user_role": "beginner"}  # ← 传入角色
)
```

**对比两种方案**：

| 方案 | 代码 | 优缺点 |
|------|------|--------|
| 用 @dynamic_prompt | `agent.invoke(..., context={"role": "expert"})` | 1个agent实例，节省资源 |
| 创建多个 agent | `expert_agent = create_agent(...)`, `beginner_agent = create_agent(...)` | 逻辑简单，占用更多内存 |

---

### 6. 结构化输出 (ToolStrategy)

```python
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model=model,
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

# Console 输出:
# name='John Doe' email='john@example.com' phone='5551234567'
print(result["structured_response"])
```

---

## Models (模型)

### 核心概念

LangChain 提供统一的模型接口，支持多种模型提供商。

---

### 1. ChatModel vs LLM

| 特性 | ChatModel | LLM |
|------|-----------|-----|
| 对话支持 | 支持多轮对话 | 单次调用 |
| 角色区分 | System/User/Assistant | 无 |
| 适用场景 | 对话应用 | 文本生成 |

---

### 2. 模型参数

```python
model = ChatOpenAI(
    model="glm-4",
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    timeout=120,          # 超时时间（秒）
    temperature=0,        # 0=确定，高=创造
    max_tokens=1000,      # 输出长度限制
    max_retries=2,        # 失败重试次数
)
```

---

### 3. 调用方式

#### invoke() - 单次调用

```python
response = model.invoke("Why do parrots have colorful feathers?")
print(response.content)
```

#### stream() - 流式输出

```python
for chunk in model.stream([HumanMessage(content="请写一首关于冬天的诗")]):
    print(chunk.content, end="", flush=True)  # 实时打印每个 token
```

#### batch() - 批量调用

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response.content)
```

#### batch_as_completed() - 并行批量（乱序到达）

```python
for idx, response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    # 结果可能乱序: (0, ...), (2, ...), (1, ...)
    print(f"[{idx}] {response.content[:50]}...")
```

---

### 4. 多轮对话

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
# Console: J'adore créer des applications.
print(response.content)
```

---

### 5. 工具调用 (Tool Calling)

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """get the weather for the city"""  # ← 必填，LLM 需要看
    return f"It's sunny in {city}."

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in Boston?")

# 查看工具调用
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")

# Console 输出:
# Tool: get_weather
# Args: {'city': 'Boston'}
```

**注意**：此时 `response.content` 为空，因为模型只是决定调用工具，还没生成最终答案。

---

### 6. 消息类型

| 类型 | 说明 |
|------|------|
| `SystemMessage` | 系统提示 |
| `HumanMessage` | 用户消息 |
| `AIMessage` | AI 响应 |
| `ToolMessage` | 工具返回结果 |

---

## Short-term Memory (短期记忆)

### 核心概念

短期记忆通过 `checkpointer` 实现线程级持久化，保存对话历史。

---

### 1. Checkpoint 机制

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

agent = create_agent(
    model,
    tools=[...],
    checkpointer=InMemorySaver()  # ← 启用短期记忆
)

config = {"configurable": {"thread_id": "1"}}

# 多次调用，对话历史会被保存
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "what's my name?"}, config)  # 记得 bob
```

---

### 2. 消息修剪 (@before_model)

长对话超出上下文窗口时，需要修剪消息。

```python
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@before_model
def trim_messages(state, runtime) -> dict | None:
    """只保留最后几条消息"""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 不处理

    # 保留第一条（系统提示）和最近几条
    first_message = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_message] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 删除所有
            *new_messages                             # 然后添加新的
        ]
    }

agent = create_agent(
    model,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[trim_messages]
)

# 注意顺序！如果把这两行换位置，AI 就不知道 name 了
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a poem about cats"}, config)
agent.invoke({"messages": "what's my name?"}, config)

# Console 输出:
# ================================== Ai Message ==================================
#
# Your name is Bob. You told me that earlier.
# If you'd like me to call you a nickname or use a different name, just say the word.
```

---

### 3. 消息总结 (SummarizationMiddleware)

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=model,              # 用于生成摘要的模型
            trigger=("tokens", 4000), # token 超过 4000 时触发
            keep=("messages", 20)     # 保留最近 20 条消息不总结
        )
    ],
    checkpointer=InMemorySaver(),
)
```

**注意**：Summarize 会永久删除旧消息，用摘要替换。Checkpoint 是快照，不是历史记录。

---

### 4. State vs Context

| 特性 | State (CustomState) | Context (CustomContext) |
|------|---------------------|-------------------------|
| 数据来源 | 继承 AgentState + 自定义 | 完全自定义 |
| 生命周期 | 跨消息持久化 | 单次请求有效 |
| 能否修改 | 可以 (Command) | 只读 |
| 访问方式 | `runtime.state["key"]` | `runtime.context.key` |
| 自动字段 | messages | 无 |

**比喻理解**：
- `Context` = SQL 查询条件 (`WHERE user_id = 'xxx'`)
- `State` = 数据库记录（查询结果）
- `Command(update=...)` = UPDATE 语句

```python
from typing import TypedDict, Any
from langgraph.types import Command
from langchain.tools import tool

class CustomState(AgentState):
    user_name: str  # 持久化

class CustomContext(BaseModel):
    user_id: str  # 请求级参数

@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState]) -> Command:
    """查询并更新用户信息"""
    user_id = runtime.context.user_id  # 从 Context 读取
    name = "John Smith" if user_id == "user_123" else "Unknown"
    return Command(update={
        "user_name": name,  # 更新 State（持久化）
        "messages": [ToolMessage("Success", tool_call_id=runtime.tool_call_id)]
    })

agent = create_agent(
    model,
    tools=[update_user_info],
    state_schema=CustomState,
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123")  # ← 传入 Context
)
```

---

### 5. 中间件执行时机

| 装饰器 | 执行时机 | 典型用途 |
|--------|----------|----------|
| `@before_model` | 每次模型调用前 | 修剪消息、过滤 |
| `@after_model` | 每次模型调用后 | 敏感词检测、日志 |
| `@after_agent` | 整个 Agent 完成后 | 最终验证、安全护栏 |

```python
@after_model
def validate_response(state, runtime) -> dict | None:
    """删除包含敏感词的消息"""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None
```

---

## Streaming (流式输出)

### 核心概念

流式输出实现逐 token 返回，提升用户体验，类似 ChatGPT 打字效果。

---

### 1. Stream Mode 类型

| mode | data 类型 | 说明 |
|------|-----------|------|
| `"updates"` | `{node: update_dict}` | 每个步骤完成后的状态更新 |
| `"messages"` | `(token, metadata)` | 逐 token 输出 |
| `"custom"` | 任意数据 | 自定义数据流 |
| `"values"` | 完整 state | 完整状态快照 |

---

### 2. Agent 进度流 (updates)

观察 Agent 执行流程，用于调试。

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather from a given city."""
    return f"The weather in {city} is sunny!"

agent = create_agent(model, tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]}
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"data: {data['messages'][-1].content_blocks}")
```

**Console 输出**：
```
step: model
data: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_...'}]
step: tools
data: [{'type': 'text', 'text': ' The weather in San Francisco is sunny! '}]
step: model
data: [{'type': 'text', 'text': 'According to my sources, the weather in San Francisco is currently sunny.'}]
```

---

### 3. Token 流 (messages)

逐 token 输出，实现 ChatGPT 打字效果。

```python
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
    stream_mode="messages"
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
```

---

### 4. 自定义数据流 (custom)

显示进度条、中间步骤等。

```python
from langgraph.config import get_stream_writer

@tool
def get_weather(city: str) -> str:
    """Get weather with progress updates."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(model, tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
    stream_mode="custom"
):
    print(chunk)
```

**Console 输出**：
```
Looking up data for city: San Francisco
Acquired data for city: San Francisco
```

---

### 5. 多模式流式

同时使用多种模式。

```python
for stream_mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
    stream_mode=["updates", "custom"]
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}\n")
```

---

### 6. 流式工具调用

```python
from langchain.messages import AIMessageChunk, AIMessage, ToolMessage

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:  # LLM 正在生成的文本
        print(token.text, end="|")
    if token.tool_call_chunks:  # 工具调用的分块
        print(token.tool_call_chunks)

def _render_completed_message(message) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

for stream_mode, data in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in Boston?"}]},
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
```

**Console 输出**：
```
[{'name': 'get_weather', 'args': '{"city":"Boston"}', 'id': 'call_...'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, ...}]
Tool response: [{'type': 'text', 'text': ' The weather in Boston is sunny! '}]
According| to| the| information| I| obtained|,| the| weather| in| Boston| is| currently| sunny|.|
```

---

### 7. 人机交互 (Human-in-the-Loop)

在 Agent 执行过程中人工审批工具调用。

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Interrupt, Command

agent = create_agent(
    model,
    tools=[get_weather],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True}  # 调用此工具时触发中断
        )
    ],
    checkpointer=InMemorySaver(),  # 必须有！
)

config = {"configurable": {"thread_id": "some_id"}}

# 第一阶段：运行到中断点
interrupts = []
for stream_mode, data in agent.stream(
    {"messages": [{"role": "user", "content": "Look up weather in Boston and SF"}]},
    config=config,
    stream_mode=["messages", "updates"],
):
    if stream_mode == "updates":
        for source, update in data.items():
            if source == "__interrupt__":
                interrupts.extend(update)
                # 显示需要审批的工具调用
                for req in update[0].value["action_requests"]:
                    print(f"Tool: {req['name']}")
                    print(f"Args: {req['args']}")

# 第二阶段：人工决策后恢复
decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {
        "decisions": [
            {"type": "approve"}  # 或 {"type": "reject"} 或 {"type": "edit", "edited_action": {...}}
        ]
    }

# 恢复执行
for stream_mode, data in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
):
    # 继续处理...
```

**决策类型**：
```python
# 1. 批准
{"type": "approve"}

# 2. 拒绝
{"type": "reject"}

# 3. 修改参数
{
    "type": "edit",
    "edited_action": {
        "name": "get_weather",
        "args": {"city": "Boston, USA"}  # 修改后的参数
    }
}
```

---

## Structured Output (结构化输出)

### 核心概念

结构化输出让 LLM 返回符合预定义 schema 的数据，便于程序处理。

---

### 1. Pydantic BaseModel 定义

```python
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email of the person")
    phone: str = Field(description="The phone number")
```

| 语法 | 含义 |
|------|------|
| `...` | Ellipsis，表示必填 |
| `description=` | 给 LLM 看的字段说明 |
| `ge=1, le=5` | 数值范围约束 |

---

### 2. Agent 结构化输出

```python
from langchain.agents import create_agent

agent = create_agent(
    model,
    response_format=ContactInfo
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# Console: name='John Doe' email='john@example.com' phone='5551234567'
```

---

### 3. OutputParser 方式

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment")
    key_points: list[str] = Field(description="Key points, lowercase, 1-3 words each")

parser = PydanticOutputParser(pydantic_object=ProductReview)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at extracting structured information.\n"
     "Respond EXCLUSIVELY with valid JSON matching this schema.\n\n"
     "{format_instructions}"),
    ("user", "Text: {input_text}")
])

chain = prompt | model | parser

result = chain.invoke({
    "input_text": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# Console: rating=5 sentiment='positive' key_points=['fast shipping', 'expensive']
```

---

### 4. 复杂 Schema 字段类型

| 字段类型 | 说明 | 示例 |
|----------|------|------|
| `str` | 字符串 | `name: str` |
| `int \| None` | 可选整数 | `rating: int \| None` |
| `Literal["a", "b"]` | 枚举值 | `sentiment: Literal["positive", "negative"]` |
| `list[str]` | 字符串列表 | `key_points: list[str]` |
| `ge=1, le=5` | 数值范围 | `rating: int = Field(..., ge=1, le=5)` |

---

## 总结

LangChain 提供了一套完整的框架来构建 AI 应用：

1. **Agent** - 智能体核心，协调工具和模型
2. **Models** - 统一的模型接口
3. **Memory** - 状态管理和对话持久化
4. **Streaming** - 流式输出提升体验
5. **Structured Output** - 可靠的结构化数据提取

通过组合这些功能，可以构建复杂的多步骤 AI 应用。
