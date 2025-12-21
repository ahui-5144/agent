from typing import Union, Annotated

from fastapi import FastAPI, Query

app = FastAPI()

# @app.get("/items")
# async def read_items(q: Union[str, None] = Query(default=None,min_length = 3, max_length=50)):
#     results = {"items" :[{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if q:
#         results.update({"q":q})
#     return results


# @app.get("/items/")
# async def read_items(
#         q: Union[str, None] = Query(
#             default=None, min_length = 1, max_length = 50, pattern="^fixedquery$"
#         )
#
# ):
#     results = {"items":[{"item_id": "Foo"},{"item_id": "Bar"}]}
#     if q:
#         results.update({"q":q})
#     return results


# 当你在使用 Query 且需要声明一个值是必需的时，只需不声明默认参数：

# @app.get("/items")
# async def read_items(q: str = Query(min_length=3)):
#     results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if q:
#         results.update({"q": q})
#     return results

# @app.get("/items")
# async def read_items(
#         q: str = Query(
#             default="fixedquery",
#             min_length=3,
#             required=False
#         )
# ):
#     results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if q:
#         results.update({"q": q})
#     return results

'''
更推荐使用Annotated（这个是python库的） Query只有fastAPI中才能用
'''
# @app.get("/items/")
# async def read_items(q: Annotated[str, Query(min_length=3)] = "fixedquery"):
#     results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if q:
#         results.update({"q": q})
#     return results

'''
查询参数列表 / 多个值
'''
# @app.get("/items")
# async def read_items(q: Annotated[list[str] | None, Query()] = None):
#     query_items = {"q": q}
#     return query_items

'''
查询参数列表 / 多个值 带默认值
'''
# @app.get("/items")
# async def read_items(q: Annotated[list[str], Query()] = ["foo", "bar"]):
#     query_item = {"q" : q}
#     return query_item

'''
声明更多元数据
'''
# @app.get("/items")
# async def read_items(
#         q: Annotated[Union[list[str], None], Query(title="Query String", description="用于搜索项目的查询字符串，支持多个值（数组形式）",min_length=3)] = None
# ):
#     results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if q:
#         results.update({"q": q})
#     return results

'''
别名参数
'''
@app.get("/items/")
async def read_items(
        q: Annotated[
            str | None,
            Query(
                alias="item_query"
            ),
        ] = None
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
