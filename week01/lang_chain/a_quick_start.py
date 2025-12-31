from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.messages.tool import tool_call
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

'''
官方文档用的是claude,其实只要是支持openai协议的都可以，我使用智谱
1.基础配置
这里使用的是智谱 AI 的模型，通过 ChatOpenAI 类来调用，因为智谱兼容 OpenAI 的 API 协议。
base_url 指定了智谱的 API 地址。
'''
llm = ChatOpenAI(
    api_key="",
    model="glm-4",  # 正确模型名：glm-4 或 glm-4-flash
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

message = [
    {"role": "user", "content": "你好"}
]

# Run the agent
# response = llm.invoke(message)
# print(response.content)



'''
system prompt
  定义了 Agent 的角色和行为：
  - 角色：爱说双关语的天气预报员
  - 可用工具：get_weather_for_location 和 get_user_location
  - 行为规则：如果用户问天气，先确认位置
'''
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


# Define context schema 上下文数据类
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """ Get weather for a location """
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

''' 
这个是claude的写法，我使用智谱的apikey
'''
# model = init_chat_model(
#     "glm-4",
#     temperature = 0
# )

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)


# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="2")
)

# print(response['structured_response'])

# 上面是claude写法，智谱使用会报错 修正：正确提取结构化响应
# 移除 response['structured_response']，改为从 response["messages"][-1].content 获取。
# 在当前 LangGraph 实现中，当使用 response_format 时，代理的最终输出会被解析并直接置于最后一条 AIMessage 的 content 字段中（类型为 ResponseFormat 实例），可直接打印或访问其属性（如 structured.punny_response）。
# LangGraph 会将最后一条 AIMessage 的 content 解析为 ResponseFormat 实例
structured = response["messages"][-1].content  # 这应该是 ResponseFormat 实例
print(structured)

# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="2")
)

structured = response["messages"][-1].content  # 这应该是 ResponseFormat 实例
print(structured)
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#

'''
执行流程

  当用户问 "what is the weather outside?" 时：

  用户 → Agent → 判断需要位置
         ↓
     调用 get_user_location() (user_id=2 → 返回 "SF")
         ↓
     调用 get_weather_for_location("SF") → 返回 "It's always sunny in SF!"
         ↓
     LLM 生成双关语回复
         ↓
     返回结构化 ResponseFormat 对象

  这个示例展示了 LangChain Agent 的核心能力：工具调用、上下文传递、结构化输出、多轮对话。
'''