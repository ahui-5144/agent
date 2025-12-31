import os
from importlib.metadata import metadata
from typing import Literal, Any

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_agent, HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessageChunk, AnyMessage,AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Interrupt, Command
from langsmith._internal._patch import request
from pydantic import BaseModel

from week01.lang_chain.c_models import model_with_tools

load_dotenv()

api_key = os.getenv("API_KEY")
ali_api_key = os.getenv("ALI_API_KEY")

model = ChatOpenAI(
    model="glm-4",
    api_key = api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# Agent progress  代理进度
@tool
def get_weather(city:str)-> str:
    """ Get weather from a given city. """
    return f" The weather in {city} is sunny! "

# agent = create_agent(
#     model=model,
#     tools=[get_weather]
# )
#
# for chunk in agent.stream(
#         {
#             "messages":[{"role": "user", "content": "What's the weather in SF?"}]
#         }
# ):
#     for step, data in chunk.items():
#         print(f"step: {step}")
#         print(f"data: {data['messages'][-1].content_blocks}")
"""
step: model
data: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_-8033029476433484437'}]
step: tools
data: [{'type': 'text', 'text': ' The weather in San Francisco is sunny! '}]
step: model
data: [{'type': 'text', 'text': 'According to my sources, the weather in San Francisco is currently sunny. Enjoy the nice weather while it lasts!'}]\
"""
"""  详解

  | 特性        | 说明                      |
  |-------------|---------------------------|
  | stream_mode | 默认（"updates"）         |
  | 返回内容    | 每个步骤完成后返回        |
  | chunk 结构  | {节点名: {状态更新数据}}  |
  | 用途        | 观察 Agent 执行流程、调试 |
"""

# LLM tokens   逐个 token输出

# agent2 = create_agent(
#     model=model,
#     tools=[get_weather]
# )
#
# for token, metadata in agent2.stream(
#     {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
#     stream_mode="messages",
# ):
#     print(f"node: {metadata['langgraph_node']}")
#     print(f"content: {token.content_blocks}")
#     print("\n")

"""  详解

  | 特性                   | 说明                                         |
  |------------------------|----------------------------------------------|
  | stream_mode="messages" | 关键参数，启用消息级流式                     |
  | 返回内容               | 每个 token 或消息块                          |
  | token                  | 当前的消息片段                               |
  | metadata               | 元数据（如节点名称）                         |
  | 用途                   | 实时显示 LLM 生成内容，类似 ChatGPT 打字效果 |

"""

# Custom updates 自定义更新
@tool
def get_weather_stream_writer(city:str)-> str:
    """" Get weather from a given city. """
    writer = get_stream_writer()
    # stream any arbitrary data  传输任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent3 = create_agent(
    model=model,
    tools=[get_weather_stream_writer],
)

# for chunk in agent3.stream(
#         {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
#             stream_mode="custom"
# ):
#     print(chunk)
""" console
Looking up data for city: San Francisco
Acquired data for city: San Francisco
"""
"""  详解

  | 组件                 | 说明                         |
  |----------------------|------------------------------|
  | get_stream_writer()  | 获取流写入器对象             |
  | writer(...)          | 发送任意数据到流中           |
  | stream_mode="custom" | 接收自定义数据               |
  | 用途                 | 显示进度条、中间步骤、日志等 |
"""

# Stream multiple modes  流式传输多种模式

# for stream_mode, chunk in agent3.stream(
#     {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
#     stream_mode=["updates", "custom"]
# ):
#     print(f"stream_mode: {stream_mode}")
#     print(f"content: {chunk}")
#     print("\n")

"""console
stream_mode: updates
content: {'model': {'messages': [AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 106, 'total_tokens': 123, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'glm-4', 'system_fingerprint': None, 'id': '2025123017071038cd2c6d42f04aeb', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019b6e82-df38-7051-899f-080880e220db-0', tool_calls=[{'name': 'get_weather_stream_writer', 'args': {'city': 'San Francisco'}, 'id': 'call_-8033144615918232058', 'type': 'tool_call'}], usage_metadata={'input_tokens': 106, 'output_tokens': 17, 'total_tokens': 123, 'input_token_details': {}, 'output_token_details': {}})]}}


stream_mode: custom
content: Looking up data for city: San Francisco


stream_mode: custom
content: Acquired data for city: San Francisco


stream_mode: updates
content: {'tools': {'messages': [ToolMessage(content="It's always sunny in San Francisco!", name='get_weather_stream_writer', id='aadabeff-4cf9-4e1b-85c7-a105a3c72eea', tool_call_id='call_-8033144615918232058')]}}


stream_mode: updates
content: {'model': {'messages': [AIMessage(content='According to the weather API, the weather in San Francisco is currently sunny. However, please note that this may not always be the case, as weather conditions can change rapidly.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 37, 'prompt_tokens': 133, 'total_tokens': 170, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'glm-4', 'system_fingerprint': None, 'id': '202512301707104cc4728fc74242e9', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b6e82-e422-7400-a698-2ee7f8120885-0', usage_metadata={'input_tokens': 133, 'output_tokens': 37, 'total_tokens': 170, 'input_token_details': {}, 'output_token_details': {}})]}}

"""

# Streaming tool calls  流式工具调用

agent4 = create_agent(
    model=model,
    tools=[get_weather],
)

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text: # LLM 正在生成的文本内容
        print(token.text, end="|")
    if token.tool_call_chunks: #  工具调用的分块信息
        print(token.tool_call_chunks)

def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls: # LLM 决定调用工具时
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage): # 工具执行完成时
        print(f"Tool response: {message.content_blocks}")

input_message = {"role":"user", "content":"What is the weather in Boston?"}

# for stream_mode, data in agent4.stream(
#         {"messages": [input_message]},
#         stream_mode=["messages", "updates"],
# ):
#     if stream_mode == "messages":
#         token, metadata = data  # 解包元组
#         if isinstance(token, AIMessageChunk):  # 类型检查
#             _render_message_chunk(token)
#     if stream_mode == "updates":
#         for source, update in data.items(): # 遍历字典
#             if source in ("model", "tools"):  # `source` captures node name
#                 _render_completed_message(update["messages"][-1])  # 取最新消息
""" console
[{'name': 'get_weather', 'args': '{"city":"Boston"}', 'id': 'call_20251230172639d0c19690b1a141ee_0', 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_20251230172639d0c19690b1a141ee_0', 'type': 'tool_call'}]
Tool response: [{'type': 'text', 'text': ' The weather in Boston is sunny! '}]
According| to| the| information| I| obtained|,| the| weather| in| Boston| is| currently| sunny|.| I| hope| this| information| is| helpful| to| you|.|
"""
"""  总结

  | 问题              | 答案                                          |
  |-------------------|-----------------------------------------------|
  | token 是什么？    | AIMessageChunk，LLM 生成的消息片段            |
  | metadata 是什么？ | dict，包含节点名、步骤等元数据                |
  | 还能解包出什么？  | 只有这两个，data = (token, metadata)          |
  | 为什么能解包？    | stream_mode="messages" 时 data 返回的就是元组 |

    # 根据 stream_mode 不同，data 的结构不同
  | stream_mode | data 类型 | data 内容           | 解包方式                         |
  |-------------|-----------|---------------------|----------------------------------|
  | "messages"  | tuple     | (token, metadata)   | token, meta = data               |
  | "updates"   | dict      | {node: update_dict} | for node, update in data.items() |
  | "custom"    | any       | 自定义数据          | 直接使用                         |
  | "values"    | dict      | 完整 state          | state = data                     |
"""

# Accessing completed messages 访问已完成的消息

class ResponseSafety(BaseModel):
    """Evaluate a response as safe or unsafe."""
    evaluation: Literal["safe", "unsafe"]

# 智谱的模型不支持推理
# safety_model = ChatOpenAI(
#     model="glm-4-flash",
#     api_key=api_key,
#     base_url="https://open.bigmodel.cn/api/paas/v4/",
#     temperature=0.7,
# )

safety_model = ChatOpenAI(
    model="qwen-max",  # 或 "qwen-qwq-32b-preview"
    api_key=ali_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@after_agent(can_jump_to=["end"]) #  can_jump_to=["end"] 允许直接跳转到 end 节点（提前结束）
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""
    stream_writer = get_stream_writer()
    # Get model response
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # Use another model to evaluate safety
    model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
    # openai写法
    # result = model_with_tools.invoke(
    #     [
    #         {
    #             "role": "system",
    #             "content":"Evaluate this AI response as generally safe or unsafe."
    #         }
    #     ],
    #     {"role":"user", "content": f"AI response: {last_message.text}"},
    # )
    # 修复：构造完整的消息列表
    evaluation_messages = [
        {
            "role": "system",
            "content": "You are a safety evaluator. Determine if the following AI response is safe or unsafe. "
                       "Respond only with the tool call using the ResponseSafety tool."
        },
        {
            "role": "user",
            "content": f"AI response to evaluate:\n\n{last_message.content}"
        }
    ]
    result = model_with_tools.invoke(evaluation_messages)
    stream_writer(result)

    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."

    return None


agent8 = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[safety_guardrail],
)

# input_message = {"role": "user", "content": "What is the weather in Boston?"}
# for stream_mode, data in agent8.stream(
#     {"messages": [input_message]},
#     stream_mode=["messages", "updates", "custom"],
# ):
#     if stream_mode == "messages":
#         token, metadata = data
#         if isinstance(token, AIMessageChunk):
#             _render_message_chunk(token)
#     if stream_mode == "updates":
#         for source, update in data.items():
#             if source in ("model", "tools"):
#                 _render_completed_message(update["messages"][-1])
#     if stream_mode == "custom":
#         # access completed message in stream
#         print(f"Tool calls: {data.tool_calls}")

"""
[{'name': 'get_weather', 'args': '{"city":"Boston"}', 'id': 'call_20251231101928fa691a37db804dfc_0', 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_20251231101928fa691a37db804dfc_0', 'type': 'tool_call'}]
Tool response: [{'type': 'text', 'text': ' The weather in Boston is sunny! '}]
According| to| the| API| call| result|,| the| weather| in| Boston| is| sunny|.| I| hope| this| answer| is| helpful| to| you|!|[{'name': 'ResponseSafety', 'args': '{"evaluation": "safe', 'id': 'call_0d80fc5b54eb4e578cbaec', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"}', 'id': '', 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'ResponseSafety', 'args': {'evaluation': 'safe'}, 'id': 'call_0d80fc5b54eb4e578cbaec', 'type': 'tool_call'}]
"""

"""
Q:@after_agen 和 @after_model有什么区别

  | 特性       | @after_model             | @after_agent           |
  |------------|--------------------------|------------------------|
  | 执行时机   | 每次模型调用后           | 整个 Agent 执行完成后  |
  | 执行次数   | 可能多次（LLM 多次调用） | 只一次（Agent 结束时） |
  | 能看到什么 | 单次 LLM 响应            | 最终完整结果           |
  | 典型用途   | 过滤敏感词、日志         | 安全护栏、最终验证     |
  | 执行位置   | model 节点之后           | agent 流程结束后       |
"""

# Streaming with human-in-the-loop

def _render_interrupt(interrupt: Interrupt) -> None:
    interrupts = interrupt.value
    for request in interrupts["action_requests"]:
        print(request["description"])

checkpointer = InMemorySaver()

agent9 = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}), # interrupt_on={"get_weather": True} 当调用 get_weather 工具时触发中断
        safety_guardrail,
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "some_id"}}

# ============ 第一阶段：初始运行，触发中断 ============
input_message = {
    "role": "user",
    "content": "Can you look up the weather in Boston and San Francisco?"
}

interrupts = []  # 用于收集中断

for stream_mode, data in agent9.stream(
    {"messages": [input_message]},  # ← 传入初始消息
    config=config,
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
            if source == "__interrupt__":
                interrupts.extend(update)
                _render_interrupt(update[0])
""" console
[{'name': 'get_weather', 'args': '', 'id': 'call_GOwNaQHeqMixay2qy80padfE', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"ci', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'ty": ', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"Bosto', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'n"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': 'get_weather', 'args': '', 'id': 'call_Ndb4jvWm2uMA0JDQXu37wDH6', 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"ci', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'ty": ', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"San F', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'ranc', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'isco"', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '}', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_GOwNaQHeqMixay2qy80padfE', 'type': 'tool_call'}, {'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_Ndb4jvWm2uMA0JDQXu37wDH6', 'type': 'tool_call'}]
Tool execution requires approval

Tool: get_weather
Args: {'city': 'Boston'}
Tool execution requires approval

Tool: get_weather
Args: {'city': 'San Francisco'}
"""


# ============ 第二阶段：人工决策后恢复执行 ============
# 根据收集到的 interrupts 生成 decisions
def _get_interrupt_decisions(interrupt: Interrupt) -> list[dict]:
    return [
        {
            "type": "edit",
            "edited_action": {
                "name": "get_weather", # tool
                "args": {"city": "Boston, Massachusetts, USA"},  # args
            },
        }
        if "boston" in request["description"].lower()
        else {"type": "approve"}# 如果是boston就修改参数，其他就直接批准
        for request in interrupt.value["action_requests"]
    ]

decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {
        "decisions": _get_interrupt_decisions(interrupt)
    }

# 恢复运行
for stream_mode, data in agent9.stream(
    Command(resume=decisions),  # ← 仅在此阶段传入 resume
    config=config,
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
            if source == "__interrupt__":
                interrupts.extend(update)
                _render_interrupt(update[0])

"""
 核心概念

  在 Agent 执行过程中，人工介入审批工具调用，可以批准、拒绝或修改工具参数。

  ---
  流程图

  用户请求 → Agent 执行 → 触发中断 → 等待审批 → 人工决策 → 恢复执行 → 返回结果
                     ↑                                    ↓
                     └───── checkpointer 保存状态 ─────────┘
                           （数据库持久化）

  ---
  关键组件

  | 组件                             | 作用                   |
  |----------------------------------|------------------------|
  | HumanInTheLoopMiddleware         | 人机交互中间件         |
  | interrupt_on={"tool_name": True} | 指定哪些工具需要审批   |
  | checkpointer                     | 必须！保存状态到数据库 |
  | thread_id                        | 会话标识，用于恢复     |
  | Command(resume=decisions)        | 恢复执行并传入决策     |
  | source == "__interrupt__"        | 捕获中断事件           |

  ---
  三个决策类型

  # 1. 批准 - 按原计划执行
  {"type": "approve"}

  # 2. 拒绝 - 不执行该工具
  {"type": "reject"}

  # 3. 修改 - 修改工具参数
  {
      "type": "edit",
      "edited_action": {
          "name": "get_weather",
          "args": {"city": "Boston, USA"}  # 修改后的参数
      }
  }

"""