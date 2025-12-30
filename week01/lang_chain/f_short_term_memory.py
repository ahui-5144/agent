import os
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, SummarizationMiddleware, dynamic_prompt, ModelRequest, after_model
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

# 加载 .env 文件中的环境变量到 os.environ
load_dotenv()

api_key = os.getenv("API_KEY")
ali_api_key = os.getenv("ALI_API_KEY")

# 查询用户信息工具
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """根据用户ID查询已存储的用户信息。

    Args:
        user_id: 要查询的用户ID，例如 "alice"、"bob" 等

    Returns:
        返回用户信息字典，如果不存在则返回 "未找到用户信息"
    """
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "未找到用户信息"

# 保存用户信息工具
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """保存用户信息到记忆存储中。

    Args:
        user_id: 用户ID，例如 "alice"、"bob" 等
        user_info: 要保存的用户信息字典，例如 {"name": "Alice", "age": 28, "city": "北京"}

    Returns:
        返回保存成功的确认消息
    """
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return f"用户 {user_id} 的信息已成功保存：{user_info}"

# 要向代理添加短期记忆（线程级持久性），需要在创建代理时指定 checkpointer
model = ChatOpenAI(
    model="glm-4",
    api_key = api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)

# aliModel = ChatOpenAI(
#     api_key=ali_api_key,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     model="qwen3-max-preview"  # 或其他可用 Qwen 模型，如 qwen-plus
# )


agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],  # 添加保存工具
    checkpointer=InMemorySaver()
)

'''
checkpoint 会存储所有对话，但生产环境通常需要配合裁剪、摘要或长期记忆方案来控制成本。
启用短期记忆后，长对话可能会超出 LLM 的上下文窗口。常见解决方案有：
1. 修剪消息：在调用LLM 之前移除前N条或后N条消息
2. 永久删除：LangGraph 状态中的消息
3. 总结消息：总结历史消息中的早期消息，并用摘要替换他们
4. 自定义策略：消息过滤等
'''

## Trim messages  修剪消息
'''
决定何时截断消息的一种方法是计算消息历史中的 token 数量，并在接近该限制时截断。
如果你使用 LangChain，可以使用 trim messages 工具，并指定要保留的 token 数量，以及 strategy （例如，保留最后一个 max_tokens ）来处理边界。
要在代理中修剪消息历史，请使用 @before_model 中间件装饰器：
'''
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """ Keep only the last few messages to fit context window. """
    messages = state["messages"]

    if len(messages) <= 3:
        return None

    first_message = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_message] + recent_messages

    return {
        "messages":[
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent2 = create_agent(
    model,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[trim_messages]
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
""" 如果把第一行和第二行换一个位置，ai就不知道name了，因为消息被裁剪了 """
# agent.invoke({"messages": "hi, my name is bob"}, config)
# agent.invoke({"messages": "write a short poem about cats"}, config)
# agent.invoke({"messages": "now do the same but for dogs"}, config)
# final_response = agent.invoke({"messages": "what's my name?"}, config)
#
# final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
checkpointer = InMemorySaver()
## Summarize messages  总结消息
agent3 = create_agent(
    model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=model,              # 用于生成摘要的模型
            trigger=("tokens", 4000), # 触发条件：token数超过4000时
            keep=("messages", 20)     # 保留：最近20条消息不总结
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
# agent3.invoke({"messages": "hi, my name is bob"}, config)
# agent3.invoke({"messages": "write a short poem about cats"}, config)
# agent3.invoke({"messages": "now do the same but for dogs"}, config)
# final_response = agent3.invoke({"messages": "what's my name?"}, config)
#
# final_response["messages"][-1].pretty_print()
#
# for i, msg in enumerate(final_response["messages"]):
#     print(f"[{i}] 类型: {msg.type}, 内容: {msg.content[:50]}...")
"""
Q:如果我使用checkpoint 那 summarize 之前的信息和之后的信息都会保存吗
A:- Checkpoint = 当前状态快照，不是历史记录
  - Summarize = 永久删除旧消息，用摘要替换
  - 如果你需要完整对话历史，需要自己实现备份机制 (使用before_model)
"""


## Access memory  访问内存
class CustomState(AgentState): # CustomState: 自定义的状态类，继承自 AgentState AgentState: LangChain 提供的基础状态类，默认包含 messages 字段
    user_id: str #  user_id: str: 扩展了一个新字段 user_id，用于在对话中传递用户ID
# 这个类定义了 Agent 运行时可以访问的数据结构

@tool # 装饰器，将函数注册为 Agent 可调用的工具
def get_user_info(runtime: ToolRuntime) -> str: # ToolRuntime: 工具运行时对象，提供访问状态的接口
    """Look up user info."""
    user_id = runtime.state["user_id"] #  runtime.state: 可以访问当前 Agent 的状态字典
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

"""
Q：runtime.state能访问哪些内容？message?然后自定义的内容？

  ┌─────────────────────────────────────────────────────────┐
  │                    CustomState                          │
  ├─────────────────────────────────────────────────────────┤
  │  继承自 AgentState:                                      │
  │  ├── messages: list[BaseMessage]   ← 消息历史            │
  │  │                                                     │
  │  自定义字段:                                             │
  │  └── user_id: str                   ← 你定义的           │
  └─────────────────────────────────────────────────────────┘
"""

# agent = create_agent(
#     model=model,
#     tools=[get_user_info], # 关键参数，指定 Agent 使用的状态模式，告诉 Agent 状态中有 user_id 字段
#     state_schema=CustomState, #  关键参数，指定 Agent 使用的状态模式，告诉 Agent 状态中有 user_id 字段
# )

# result = agent.invoke({
#     "messages": "look up user information",
#     "user_id": "user_123",
# })
#
# print(result["messages"][-1].content) #According to the information we have queried, the user's name is John Smith.


# Write short-term memory from tools   从工具中写入短期记忆

class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState]) -> Command :
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name, # # 更新 State 字段（持久化）
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

# agent5 = create_agent(
#     model=model,
#     tools=[update_user_info,greet],
#     state_schema=CustomState,
#     context_schema=CustomContext,
# )
#
# result = agent5.invoke(
#     {"messages": [{"role": "user", "content": "greet the user"}]},
#     context=CustomContext(user_id="user_123"),
# )
# print(result) # 输出全部内容
# print(result["messages"][-1].content) # 输出最后的结果
"""
 执行流程示意

  用户: "打个招呼"
     ↓
  LLM: 决定调用 greet 工具
     ↓
  greet 执行: 发现 user_name is None
     ↓
  返回 ToolMessage: "Please call the 'update_user_info' tool..."
     ↓
  LLM: 收到这个消息，理解需要先调用 update_user_info
     ↓
  LLM: 决定调用 update_user_info 工具
  
  
runtime.tool_call_id
  | 属性   | 说明                                    |
  |--------|-----------------------------------------|
  | 来源   | 系统自动生成，在 LLM 决定调用工具时分配 |
  | 作用   | 标识"这是哪一次工具调用"                |
  | 用途   | 让 ToolMessage 能正确对应到原始调用     |
  | 必须性 | 必须传入，否则消息关联会出错            |
  

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                              执行流程                                        │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │  ① HumanMessage         → "greet the user"                                  │
  │        ↓                                                                    │
  │  ② AIMessage (tool_calls) → 调用 greet, id='call_-803309...'                  │
  │        ↓                                                                        │
  │  ③ ToolMessage          → "Please call update_user_info..."                   │
  │        ↓                                                                        │
  │  ④ AIMessage (tool_calls) → 调用 update_user_info, id='call_-803307...'        │
  │        ↓                                                                        │
  │  ⑤ ToolMessage          → "Successfully looked up user information"            │
  │        ↓                                                                        │
  │  ⑥ AIMessage (tool_calls) → 再次调用 greet, id='call_-803314...'               │
  │        ↓                                                                        │
  │  ⑦ ToolMessage          → "Hello John Smith!"                                  │
  │        ↓                                                                        │
  │  ⑧ AIMessage (final)    → "Hello John Smith! How can I assist you today?"      │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘  
"""
"""
runtime.state 和 runtime.context 有什么区别
  核心区别

  | 特性       | runtime.state                                    | runtime.context               |
  |------------|--------------------------------------------------|-------------------------------|
  | 数据来源   | CustomState 类                                   | CustomContext 类              |
  | 生命周期   | 跨消息持久化                                       | 单次请求有效                  |
  | 能否修改   | ✅ 可以（通过 Command(update={...})）            | ❌ 不能（只读）               |
  | 访问方式   | runtime.state["key"] 或 runtime.state.get("key") | runtime.context.key（点语法） |
  | 继承内容   | 自动包含 messages                                | 无继承，完全自定义            |
  | 创建时指定 | state_schema=CustomState                         | context_schema=CustomContext  |
  
  
  | 问题                          | 答案                                             |
  |-------------------------------|--------------------------------------------------|
  | 为什么 user_id 放在 Context？ | 因为它是请求级参数，每次调用时传入，不需要持久化 |
  | 为什么 user_name 放在 State？ | 因为它是查询结果，需要持久化供后续消息使用       |
  | 访问语法为什么不同？           | Context 用点语法（属性访问），State 用字典语法   |
  | 哪些字段自动有？              | State 自动有 messages，Context 完全自定义        |  
  
 CustomState和 CustomContext的区别
 > CustomContext就代表用户输入的信息，类似于查询信息。CustomState就类似数据库信息，因为你查询信息不同，数据库查出来的答
案自然也不一样。然后Command(update 就类似根据查询信息更新数据库中的值，是这个意思吗

● 完全正确！ 你的理解非常准确 👍
  ---
  你的比喻总结

  | 概念                  | 比喻               | 实际含义                   |
  |-----------------------|--------------------|----------------------------|
  | CustomContext         | 查询条件（输入）   | WHERE user_id = 'user_123' |
  | CustomState           | 数据库记录（数据） | 查询结果、会话状态         |
  | Command(update={...}) | UPDATE 语句        | 更新 State 中的值          |

"""

# Prompt  提示 访问短期记忆（状态）以在中间件中根据对话历史或自定义状态字段创建动态提示。

class CustomContext2(TypedDict):
    user_name: str

def get_weather(city: str) -> str:
    """ Get the weather for the city. """
    return f"The weather in {city} is sunny!"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

agent6 = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext2,
)

result = agent6.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext2(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()

"""
================================ Human Message =================================

What is the weather in SF?
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_-8032986767277860076)
 Call ID: call_-8032986767277860076
  Args:
    city: San Francisco
================================= Tool Message =================================
Name: get_weather

The weather in San Francisco is sunny!
================================== Ai Message ==================================

The weather in San Francisco is sunny, John Smith!
"""

# Before model  在模型之前  在 @before_model 中间件中访问短期记忆（状态），以在模型调用之前处理消息。
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


agent = create_agent(
    model=model,
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()

@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """ Remove messages containing sensitive words. """
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None