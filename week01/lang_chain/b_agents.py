from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from pydantic import tools

# static model
model = ChatOpenAI(
    model="glm-4",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)

agent = create_agent(model, tools=tools)

# dynamic model  @wrap_model_call
basic_model = ChatOpenAI(model="glm-4-flash")
advanced_model = ChatOpenAI(model="glm-4")

def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])  # 获取对话轮数
    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynamic_model_selection]
)


