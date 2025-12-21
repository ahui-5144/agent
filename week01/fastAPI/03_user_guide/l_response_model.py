from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: list[str] = []

# @app.post("/items/")
# async def read_items(item: Item) -> Item:
#     return item
#
# @app.get("/items/")
# async def read_items() -> list[Item]:
#     return [
#         Item(name="Portal Gun", price=42.0),
#         Item(name="Plumbus", price=32.0),
#     ]


# @app.post("/items/", response_model=Item)
# async def create_item(item: Item) -> Any:
#     return item
#
# @app.get("/items/", response_model=list[Item])
# async def read_items() -> Any:
#     return [
#         {"name": "Portal Gun", "price": 42.0},
#         {"name": "Plumbus", "price": 32.0},
#     ]

# class UserIn(BaseModel):
#     username: str
#     password: str
#     email: EmailStr
#     full_name: str | None = None
#
# class UserOut(BaseModel):
#     username: str
#     email: EmailStr
#     full_name: str | None = None

# @app.post("/user/")
# async def create_user(user: UserIn) -> UserOut:
#     return user

'''
上述写法字段存在冗余
'''
class UserBase(BaseModel):
    name: str
    email: EmailStr
    full_name: str | None = None

class UserIn(UserBase):
    password: str

@app.post("/user/")
async def create_user(user: UserIn) -> UserBase:
    return user

# @app.get("/portal")
# async def get_portal(teleport: bool = False) -> Response:
#     if teleport:
#         return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
#     return JSONResponse(content={"message": "Here's your interdimensional portal."})


# @app.get("/portal")
# async def get_portal(teleport: bool = False) -> Response | dict:
#     if teleport:
#         return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
#     return {"message": "Here's your interdimensional portal."}



class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float = 10.5
    tags: list[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: str):
    return items[item_id]

'''
response_model_exclude_unset = true 默认值不会填充，只会返回有值的数据
'''