import os
from typing import Generic, Union, Callable, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import SchemaT
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")


model = ChatOpenAI(
    model="glm-4",
    api_key= api_key,
    base_url= base_url,
)


class ContractInfo(BaseModel):
    """ Contract info for a person. """
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email of the person ")
    phone: str = Field(description="The phone number of the person")


# agent = create_agent(
#     model=model,
#     response_format=ContractInfo,
# )
#
# result = agent.invoke({
#     "messages": [{"role": "user","content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
# })
# print(result["structured_response"]) # name='John Doe' email='john@example.com' phone='5551234567'
#

class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")


parser = PydanticOutputParser(pydantic_object=ProductReview)

# 构建严格的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at extracting structured contact information.\n"
     "Extract the name, email, and phone number from the given text.\n"
     "Respond EXCLUSIVELY with valid JSON matching this schema. "
     "Do NOT include any explanation, apology, or additional text.\n\n"
     "{format_instructions}"),
    ("user", "Text: {input_text}")
])

# 构建链：提示 → 模型 → 解析器
chain = prompt | model | parser

# 调用
result = chain.invoke({
    "input_text": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'",
    "format_instructions": parser.get_format_instructions()
})

# 直接打印结构化对象（Pydantic 实例）
print(result) # rating=5 sentiment='positive' key_points=['fast shipping', 'expensive']