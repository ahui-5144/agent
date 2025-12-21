from fastapi import FastAPI, Path, Body

from typing import Annotated
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str  # 表示name是必填
    description: str | None = None
    price: float
    tax: float | None = None


# @app.put("/items/{item_id}")
# async def update_item(
#         item_id: Annotated[int, Path(title="The ID of the item to get", ge=0, le=1000)],
#         q: str | None = None,
#         item: Item | None = None,
# ):
#     results = {"item_id": item_id}
#     if q:
#         results.update({"q": q})
#     if item:
#         results.update({"item": item})
#     return results


'''
多个请求体参数
'''


class User(BaseModel):
    username: str
    full_name: str | None = None

#
# @app.put("/items/{item_id}")
# async def read_item(item_id: int, item: Item, user: User):
#     results = {"item_id": item_id, "item": item, "user": user}
#     return results


'''
请求体中的单一值


Body 的具体作用总结：
标记参数来源：明确参数从 POST/PUT/PATCH 等请求的「请求体」（JSON / 表单等）读取，而非 URL 中的路径 / 查询参数；
兼容基础类型请求体：解决「基础类型（int/str）默认解析为查询参数」的问题，强制其从请求体读取；
多请求体参数合并：FastAPI 会将多个 Body 参数（包括隐式的 BaseModel）合并为一个 JSON 请求体（而非多个请求体）；
附加校验 / 元数据：可通过 Body(ge=1, le=10, description="重要性") 给请求体参数加校验规则 / 文档描述（类似 Query/Path）。


Body 核心作用：标记参数从 HTTP 请求体 读取，类比 Java 的 @RequestBody；
关键规则：
BaseModel 类型参数：默认隐式为 Body 参数（无需显式加 Body()）；
基础类型（int/str/bool）：必须显式加 Body() 才会从请求体读取（否则为查询参数）；
'''

@app.put("/items/{item_id}")
async def update_item(
        item_id: int,
        item: Item,
        user: User,
        importance: Annotated[int, Body()]
):
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results
