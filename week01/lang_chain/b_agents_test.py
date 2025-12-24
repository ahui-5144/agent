"""
测试 Dynamic Model 中间件

运行前确保设置环境变量：
export OPENAI_API_KEY="your-api-key"
或者直接在代码中设置
"""

import os
import sys
# 设置 UTF-8 编码输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# ==================== 配置 ====================
os.environ["OPENAI_API_KEY"] = "1d8df09fb3034f5d9a5740bf51efef8f.nuZBo5BuGT3A70Ps"  # 智谱API key

# ==================== 定义两个模型 ====================
basic_model = ChatOpenAI(
    model="glm-4-flash",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)

advanced_model = ChatOpenAI(
    model="glm-4",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)


# ==================== 定义工具 ====================
@tool
def calculate(x: int, y: int) -> str:
    """简单计算工具"""
    return f"结果: {x + y}"


# ==================== 定义中间件（带日志） ====================
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def dynamic_model_middleware(request: ModelRequest, next_fn):
    """根据对话复杂度选择模型"""
    messages = request.state["messages"]
    message_count = len(messages)

    # 打印日志，观察模型切换
    print(f"[DEBUG MIDDLEWARE] 当前对话轮数: {message_count}")

    # 选择模型
    if message_count > 3:
        selected = advanced_model
        print(f"[DEBUG MIDDLEWARE] 使用高级模型: glm-4")
    else:
        selected = basic_model
        print(f"[DEBUG MIDDLEWARE] 使用基础模型: glm-4-flash")

    # 覆盖模型并调用
    return next_fn(request.override(model=selected))


# ==================== 创建 Agent ====================
checkpointer = InMemorySaver()

agent = create_agent(
    model=basic_model,  # 默认模型
    tools=[calculate],
    middleware=[dynamic_model_middleware],
    checkpointer=checkpointer,
)


# ==================== 测试函数 ====================
def safe_print(content):
    """安全打印，处理 emoji"""
    print(content)

def test_dynamic_model():
    """测试模型动态切换"""

    # 使用相同的 config 保持对话历史
    config = {"configurable": {"thread_id": "test-thread-1"}}

    print("=" * 50)
    print("测试 1: 简单对话（轮数 <= 3，使用 flash 模型）")
    print("=" * 50)

    for i in range(1, 4):
        print(f"\n--- 第 {i} 轮对话 ---")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": f"这是第{i}次提问，你好"}]},
            config=config,
        )
        content = response['messages'][-1].content[:50]
        safe_print(f"回复: {content}...")

    print("\n" + "=" * 50)
    print("测试 2: 复杂对话（轮数 > 3，切换到 glm-4）")
    print("=" * 50)

    for i in range(4, 7):
        print(f"\n--- 第 {i} 轮对话 ---")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": f"这是第{i}次提问，帮我分析一下"}]},
            config=config,
        )
        content = response['messages'][-1].content[:50]
        safe_print(f"回复: {content}...")

    print("\n" + "=" * 50)
    print("测试 3: 新对话（重置对话，重新用 flash）")
    print("=" * 50)

    # 使用新的 thread_id 重置对话
    new_config = {"configurable": {"thread_id": "test-thread-2"}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "新对话开始"}]},
        config=new_config,
    )
    content = response['messages'][-1].content[:50]
    safe_print(f"回复: {content}...")


# ==================== 测试场景2：根据请求长度选择 ====================
class LengthBasedMiddleware(AgentMiddleware):
    """根据用户输入长度选择模型的中间件"""

    def __init__(self, basic_model, advanced_model, threshold=50):
        self.basic_model = basic_model
        self.advanced_model = advanced_model
        self.threshold = threshold

    def __call__(self, request: ModelRequest, handler) -> ModelResponse:
        messages = request.state["messages"]
        last_message = messages[-1].content if messages else ""
        length = len(last_message)

        print(f"[DEBUG] 输入长度: {length} 字符")

        if length > self.threshold:
            selected = self.advanced_model
            print(f"[DEBUG] 输入较长，使用高级模型")
        else:
            selected = self.basic_model
            print(f"[DEBUG] 输入较短，使用基础模型")

        return handler(request.override(model=selected))


def create_length_based_agent():
    """根据用户输入长度选择模型的 Agent"""
    return create_agent(
        model=basic_model,
        tools=[calculate],
        middleware=[LengthBasedMiddleware(basic_model, advanced_model, threshold=50)],
    )


def test_length_based():
    """测试基于长度的模型选择"""
    agent = create_length_based_agent()

    print("\n" + "=" * 50)
    print("测试: 短输入 vs 长输入")
    print("=" * 50)

    print("\n--- 短输入 ---")
    agent.invoke({"messages": [{"role": "user", "content": "你好"}]})

    print("\n--- 长输入 ---")
    long_text = "你好" * 20  # 80+ 字符
    agent.invoke({"messages": [{"role": "user", "content": long_text}]})


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("开始测试 Dynamic Model 中间件\n")

    # 测试1: 基于对话轮数
    test_dynamic_model()

    # 测试2: 基于输入长度
    # test_length_based()
