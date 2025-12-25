import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import tools, BaseModel

# 加载 .env 文件
load_dotenv()

# 读取环境变量
api_key = os.getenv("API_KEY")

# static model 静态模型在创建代理时配置一次，并在整个执行过程中保持不变。这是最常见和直接的方法。
model = ChatOpenAI(
    model="glm-4",
    api_key = api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)

agent = create_agent(model, tools=[])

# dynamic model  @wrap_model_call 该中间件在请求中修改模型
basic_model = ChatOpenAI(model="glm-4-flash", api_key=api_key)
advanced_model = ChatOpenAI(model="glm-4", api_key=api_key)

def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])  # 获取对话轮数
    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model=model))

# agent = create_agent(
#     model=basic_model,
#     api_key=api_key,
#     tools=[],
#     middleware=[dynamic_model_selection]
# )
'''
Q:上述方法在对话轮次大于10轮时切换模型，那上下文信息回丢失吗？
A:这种动态切换模型的实现方式不会丢失上下文，只是让不同的模型处理相同的对话历史。这是 LangChain agent 中间件的一种推荐用法
'''


# 将工具列表传递给代理
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


agent2 = create_agent(model, tools=[search, get_weather])


# Tool error handling  工具错误处理
# 当工具失败时，代理将返回一个 ToolMessage ，其中包含自定义错误消息：
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

'''
Q:这样是不是就可以在大模型运行期间，查询自己的数据库（比如用户的学习记录，或者是其他的东西，又或者查询互联网之内的）？

 通过自定义工具，agent 可以在运行期间：

  | 功能         | 实现方式                                                               |
  |--------------|------------------------------------------------------------------------|
  | 查询数据库   | 在工具函数中连接 MySQL/PostgreSQL/MongoDB 等，查询用户学习记录、订单等 |
  | 调用内部 API | 发送 HTTP 请求到自己的业务系统                                         |
  | 搜索互联网   | 集成 Google Search、Bing API 或爬虫                                    |
  | 读写文件     | 读取本地文档、保存生成的内容                                           |
  | 调用其他 LLM | 组合多个模型的能力                                                     |
工具函数让 LLM 从"只有训练数据的封闭系统"变成了"可以实时获取外部信息的智能体"。
'''

# System prompt  系统提示   你可以通过提供提示来塑造你的代理如何处理任务
# agent3 = create_agent(
#     model,
#     tools,
#     system_prompt="You are a helpful assistant. Be concise and accurate."
# )

''' 下面是claude特有的特有的功能 '''
literary_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type": "text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    )
)

# result = literary_agent.invoke(
#     {"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]}
# )

# Dynamic system prompt  动态系统提示
class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent4 = create_agent(
    model= model,
    # tools=[web_search], 这个需要安装依赖，暂时用不上
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
# result = agent4.invoke(
#     # {"messages": [{"role": "user", "content": "Explain machine learning"}]},
#     {"messages": [HumanMessage("Explain machine learning")]},
#     context={"user_role": "beginner"}
# )
#
# structured = result["messages"][-1].content  # 这应该是 ResponseFormat 实例
# print(structured)

'''
Q:这个@dynamic_prompt 动态系统提示，是不是就是根据上下文信息动态切换 prompt?这个的好处是什么?我不是可以先获取上下文直接生成不同的prompt吗？
还是说不同的prompt需要创建新的agent，导致浪费大量的系统资源


 @dynamic_prompt 的作用

  @dynamic_prompt
  def user_role_prompt(request: ModelRequest) -> str:
      user_role = request.runtime.context.get("user_role", "user")
      # 根据 context 动态返回不同的 prompt
      if user_role == "expert":
          return "Provide detailed technical responses."
      elif user_role == "beginner":
          return "Explain concepts simply."
      return "You are a helpful assistant."

  你的两种方案对比

  | 方案                         | 代码                                               | 优缺点                                             |
  |------------------------------|----------------------------------------------------|----------------------------------------------------|
  | 方案A：用 @dynamic_prompt    | agent.invoke(..., context={"user_role": "expert"}) | ✅ 一个 agent 实例✅ 逻辑封装在内部❌ 稍微复杂一点 |
  | 方案B：外部直接传不同 prompt | 创建多个 agent，每个不同 prompt                    | ❌ 多个 agent 实例✅ 逻辑简单直观                  |

  核心好处

  你说得对，不是必须用 @dynamic_prompt。它的好处是：

  不用 @dynamic_prompt：
  ─────────────────────────────────────────────
  expert_agent = create_agent(model, system_prompt="...expert...")
  beginner_agent = create_agent(model, system_prompt="...beginner...")

  → 创建了 2 个 agent 实例，占用 2 倍内存

  用 @dynamic_prompt：
  ─────────────────────────────────────────────
  agent = create_agent(model, middleware=[user_role_prompt])

  agent.invoke(..., context={"user_role": "expert"})     # 动态生成
  agent.invoke(..., context={"user_role": "beginner"})   # 动态生成

  → 只有 1 个 agent 实例，prompt 在运行时生成

  总结

  - 小项目：直接在外层创建不同 prompt 的 agent 更简单
  - 大项目：用 @dynamic_prompt 可以复用同一个 agent 实例，节省资源，逻辑更内聚
'''



# ToolStrategy 使用人工工具调用生成结构化输出
class ContactInfo(BaseModel):
    name:str
    email:str
    phone:str

agent5 = create_agent(
    model= model,
    response_format=ToolStrategy(ContactInfo)
)

result = agent5.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

#  获取 agent 的结构化输出（如果配置了 response_format）
print(result["structured_response"])
# console name='John Doe' email='john@example.com' phone='5551234567'
