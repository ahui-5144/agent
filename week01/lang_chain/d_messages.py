import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


# 加载 .env 文件中的环境变量到 os.environ
load_dotenv()

api_key = os.getenv("API_KEY")
model = ChatOpenAI(
    api_key=api_key,
    model="glm-4",  # 需要指定模型
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    timeout=120, # 超时时间
    temperature=0, # 控制模型输出的随机性。较高的数量使回答更具创造性；较低的数量使回答更确定。
    max_tokens=1000, # 有效控制输出可以有多长
    max_retries=2, # 如果由于网络超时或速率限制等问题导致请求失败，系统将尝试重新发送请求的最大次数。
)

# Basic usage 基本用法
system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# Use with chat models
# messages = [system_msg, human_msg]
# response = model.invoke(messages)
# print(response)

# Text prompts  文本提示   文本提示是字符串——适用于不需要保留对话历史的简单生成任务。

# Message prompts  消息提示

# messages = [
#     SystemMessage("You are a poetry expert"),
#     HumanMessage("Write a haiku about spring"),
#     AIMessage("Cherry blossoms bloom...")
# ]
# response = model.invoke(messages)
# print(response)

# Dictionary format  字典格式
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
print(response)