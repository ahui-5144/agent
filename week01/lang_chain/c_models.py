import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import BaseModel, Field

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


# response = model.invoke("Why do parrots have colorful feathers?")
# print(response)

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant that translates English to French."},
#     {"role": "user", "content": "Translate: I love programming."},
#     {"role": "assistant", "content": "J'adore la programmation."},
#     {"role": "user", "content": "Translate: I love building applications."}
# ]
#
# response = model.invoke(conversation)
# print(response)  # AIMessage("J'adore créer des applications.")

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

# response = model.invoke(conversation)
# print(response.content)  # AIMessage("J'adore créer des applications.")

'''
Q:聊天模型和llm有什么区别？
A:聊天模型支持多轮对话、角色区分 llm单次调用
'''

# stream  可以实现逐token输出，提升用户体验
# for chunk in model.stream([HumanMessage(content="请写一首关于冬天的诗")]):
#     print(chunk.content, end="", flush=True)  # 实时打印每个 token

# Batch

# responses = model.batch([
#     "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ])
# for response in responses:
#     print(response)

# for response in model.batch_as_completed([
#     "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ]):
#     print(response)
'''
在使用 batch_as_completed() 时，结果可能会乱序到达。每个结果都包含输入索引，用于匹配并按需重建原始顺序。
(0, AIMessage(content="Parrots have colorful feathers primarily for communication and camouflage purposes.\n\nCommunication: Colorful plumage is an important means of communication among parrots. It helps them recognize members of their own species, find mates, and establish social hierarchies within their flocks. Bright colors can signal health, vitality, and genetic fitness, making them more attractive to potential mates. Additionally, certain behaviors, such as displays during courtship, are enhanced by the vibrant colors, which can also be used to threaten competitors or predators.\n\nCamouflage: While not all parrots rely on camouflage, many do have colors that help them blend into their natural habitats. For example, some forest-dwelling parrots have green feathers that allow them to merge with the foliage, making it harder for predators to spot them. Similarly, some parrots have patterns that break up their outline, providing additional concealment.\n\nIt's also worth noting that the evolution of colorful feathers in parrots may have been influenced by sexual selection, where the preference for bright, attractive traits by mates drives the development of those traits over time.\n\nLastly, the pigments responsible for the colors can have a role in protecting the birds from ultraviolet (UV) radiation. Some of the pigments吸收UV light, which may help prevent damage to the bird's skin and DNA.\n\nOverall, the colorful feathers of parrots serve multiple functions, including communication, camouflage, and protection, which have contributed to their evolutionary success.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 294, 'prompt_tokens': 13, 'total_tokens': 307, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'glm-4', 'system_fingerprint': None, 'id': '2025122517424492ce05264f0041a1', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b54e3-a702-7b12-84cc-831a7a48a59c-0', usage_metadata={'input_tokens': 13, 'output_tokens': 294, 'total_tokens': 307, 'input_token_details': {}, 'output_token_details': {}}))
(2, AIMessage(content='Quantum computing is a type of computing that utilizes the principles of quantum mechanics to process information. Unlike classical computing, which is based on binary digits or bits that can have a value of either 0 or 1, quantum computing uses quantum bits or qubits that can exist in multiple states at the same time, thanks to a quantum phenomenon called superposition.\n\nSuperposition allows a qubit to be in two states (0 and 1) simultaneously, which means that a quantum computer with multiple qubits can perform many calculations in parallel. This property enables quantum computers to solve certain problems much faster than classical computers.\n\nAnother important quantum mechanical principle used in quantum computing is entanglement. Entanglement is a correlation between two or more qubits, where the state of one qubit cannot be described independently of the state of the others, even if they are separated by a large distance. This property allows for the transfer of information across large distances without any loss of time, as would be experienced in classical communication.\n\nQuantum computers are not intended to replace classical computers for all tasks, but they have the potential to revolutionize fields that require complex calculations, such as:\n\n1. Cryptography: Quantum computers could break many of the encryption algorithms currently in use, necessitating the development of new quantum-resistant encryption methods.\n2. Material science: Quantum computers could simulate the behavior of molecules and atoms, leading to the discovery of new materials and drugs.\n3. Optimization problems: Quantum computers could solve complex optimization problems more efficiently than classical computers, which would have applications in logistics, finance, and artificial intelligence.\n4. Machine learning: Quantum computing could potentially speed up the training of machine learning models and improve pattern recognition.\n\nHowever, quantum computing is still in its infancy, and practical quantum computers that can outperform classical computers for real-world applications are yet to be developed. Challenges include the issue of qubit stability, error correction, and the development of algorithms that can effectively utilize the power of quantum computing.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 401, 'prompt_tokens': 10, 'total_tokens': 411, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'glm-4', 'system_fingerprint': None, 'id': '2025122517424456b7deea79e24b15', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b54e3-a703-7c33-ba41-81f65f05fff7-0', usage_metadata={'input_tokens': 10, 'output_tokens': 401, 'total_tokens': 411, 'input_token_details': {}, 'output_token_details': {}}))
(1, AIMessage(content="The principle of how airplanes fly is based on the science of aerodynamics. Here's a simplified explanation:\n\n1. **Wings (Airfoils):** The shape of an airplane's wings is crucial for generating lift. Wings are typically designed as airfoils, which means they are shaped to create a pressure difference when air flows over and under them. The top surface of the wing is curved, while the bottom is flatter. This design causes the air to move faster over the top of the wing than the bottom due to the shape and angle of attack (the angle between the wing and the oncoming airflow).\n\n2. **Bernoulli's Principle:** As the air moves faster over the top of the wing, it creates lower pressure according to Bernoulli's principle. Simultaneously, the air moving slower under the wing creates higher pressure. This pressure difference generates an upward force called lift.\n\n3. **Angle of Attack:** The angle of attack can be adjusted to increase or decrease lift. A higher angle of attack generates more lift but also more drag (air resistance). An airplane increases the angle of attack to take off and climb, and decreases it to descend.\n\n4. **Thrust:** While lift is essential for an airplane to overcome gravity, thrust is necessary to overcome drag and move the airplane forward through the air. Thrust is provided by the engines, typically jet engines or propellers. Jet engines compress incoming air, mix it with fuel, and ignite it to create a high-speed exhaust that pushes the airplane forward. Propellers, on the other hand, act like rotating wings to generate thrust.\n\n5. **Control Surfaces:** To control the flight, airplanes have control surfaces such as the elevator (for pitch), ailerons (for roll), and rudder (for yaw). These surfaces can be moved to change the airflow over the wings and tail, allowing the pilot to maneuver the aircraft.\n\n6. **Stability and Balance:** Airplanes are designed to maintain stability and balance in flight. The distribution of weight, the position of the wings and tail, and the design of the fuselage all contribute to making the aircraft controllable and safe.\n\nIn summary, airplanes fly by using their wings to generate lift, engines to provide thrust, and control surfaces to manage the direction and stability of the aircraft. It's a complex interaction of forces that has been refined through years of scientific research and engineering development.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 492, 'prompt_tokens': 10, 'total_tokens': 502, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'glm-4', 'system_fingerprint': None, 'id': '20251225174244d6bce7b63dc64d9a', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b54e3-a703-7c33-ba41-81e16124834f-0', usage_metadata={'input_tokens': 10, 'output_tokens': 492, 'total_tokens': 502, 'input_token_details': {}, 'output_token_details': {}}))

'''

list_of_inputs = [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]
# response = model.batch(
#                 list_of_inputs,
#                 config={
#                     'max_concurrency': 5,  # Limit to 5 parallel calls  最大并发5
#                 }
#             )
# print(response)


# Tool calling 模型可以请求调用执行从数据库获取数据、搜索网络或运行代码等任务的工具

@tool
def get_weather(city: str)-> str:
    ''' get the weather for the city ''' #这个是必填的，不然LLM就不知道这个工具是干什么的
    return f"It's sunny in {city}."


model_with_tools = model.bind_tools([get_weather])

# response = model_with_tools.invoke("What's the weather Boston?")
# for tool_call in response.tool_calls:
#     # View tool calls made by the model
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")
# Tool: get_weather
# Args: {'city': 'Boston'}
# print(response.content) # 这里返回为空，因为这里只是大模型调用工具，并没有生成答案


# Structured output  结构化输出  glm-4不支持 with_structured_output
class Movie(BaseModel):
    ''' A movie with details.'''
    title:str = Field(..., description="Title of the movie")
    year:int = Field(...,description="The year the movie was released")
    director:str = Field(..., description="The director of the movie")
    rating:float = Field(..., description="The movie's rating out of 10")

# model_with_structure = model.with_structured_output(Movie)
# response = model_with_structure.invoke("Provide details about the movie Inception")
# print(response)


'''
  title: str = Field(..., description="Title of the movie")

  | 部分         | 含义                                              |
  |--------------|---------------------------------------------------|
  | ...          | Ellipsis 对象，表示该字段是必填的（required）     |
  | description= | 给 LLM 的描述说明，告诉模型这个字段应该填什么内容 | 
    简记: ... = "必须有值"，description = "给 LLM 看的说明"
'''

# parser = JsonOutputParser(pydantic_object=Movie)
#
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个电影信息助手。"),
#     ("user", "{query}\n\n{format_instructions}")
# ])

# chain = prompt | model | parser

# response = chain.invoke({
#     "query": "Provide details about the movie Inception",
#     "format_instructions": parser.get_format_instructions()
# })
'''
BaseModel 定义 → get_format_instructions() → 格式说明 → LLM 按格式返回
你定义一次 Movie 类，get_format_instructions() 自动把要求翻译给 LLM 听。
'''
# print(response)


# Multimodal  多模态  glm-4是纯文本
# response = model.invoke("Create a picture of a cat")
# print(response.content_blocks)


# Reasoning  推理 (GLM-4 不支持 reasoning blocks，这是 OpenAI o1 系列的特性)
# 普通流式输出
# if __name__ == "__main__":
#     for chunk in model.stream("Why do parrots have colorful feathers?"):
#         print(chunk.content, end="", flush=True)