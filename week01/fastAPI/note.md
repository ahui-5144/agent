# FastAPI 学习笔记

## 1. 快速开始 (Quick Start)

### 1.1 Python 类型注解

Python 的类型注解让代码更清晰，IDE 能提供更好的提示。

```python
# 基本类型注解
def get_full_name(first_name: str, last_name: str):
    full_name = first_name.title() + " " + last_name.title()
    return full_name

print(get_full_name("john", "wick"))  # John Wick

# 多个类型注解
def get_name_with_age(name: str, age: int):
    return f"{name.title()} is {age} years old"

# 常用类型
def get_items(item_a: str, item_b: int, item_c: float, item_d: bool, item_e: bytes):
    return item_a, item_b, item_c, item_d, item_e
```

| 类型 | 说明 | 示例值 |
|------|------|--------|
| `str` | 字符串 | `"hello"` |
| `int` | 整数 | `42` |
| `float` | 浮点数 | `3.14` |
| `bool` | 布尔值 | `True`/`False` |
| `bytes` | 字节 | `b"hello"` |

---

### 1.2 复杂类型注解 (typing 模块)

```python
from typing import List, Tuple, Set, Dict

# 列表
def process_items(items: List[str]):
    for item in items:
        print(item.title())

process_items(['apple', 'banana', 'cherry'])

# 元组（固定长度）
def process_items(items_t: Tuple[str, int, float]):
    for item in items_t:
        print(item)

process_items(("张三", 12, 13.5))

# 字典
def process_items(prices: Dict[str, float]):
    for item_name, item_price in prices.items():
        print(item_name, item_price)

process_items({"苹果": 5.99, "香蕉": 3.50})

# 自定义类
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

def get_person_info(one_person: Person):
    return f"{one_person.name} {one_person.age}"

one_person = Person("John", 20)
print(get_person_info(one_person))  # John 20
```

---

### 1.3 Pydantic 模型

Pydantic 是数据验证库，FastAPI 使用它来定义请求/响应模型。

```python
from datetime import datetime
from pydantic import BaseModel

class User(BaseModel):
    id: int                           # 必填
    name: str = "john wick"           # 有默认值，可选
    signup_ts: datetime | None = None # 可选，默认 None
    friends: list[int] = []           # 默认空列表

# 使用
external_data = {
    "id": "123",                      # 自动转换为 int
    "signup_ts": datetime.now(),
    "friends": [1, "2", b"3"],       # 自动转换为 [1, 2, 3]
}

user = User(**external_data)
print(user)
# Console: id=123 name='john wick' signup_ts=datetime.datetime(2025, 12, 16, ...) friends=[1, 2, 3]
```

**Pydantic 特性**：
- 自动类型转换
- 自动验证
- 提供 `.model_dump()` 方法转换为字典

---

## 2. 异步编程 (Async/Await)

FastAPI 支持异步函数，能处理高并发请求。

```python
import asyncio

async def get_burgers(number: int):
    for i in range(number):
        await asyncio.sleep(1)  # 模拟异步操作
    return number
```

| 关键字 | 说明 |
|--------|------|
| `async def` | 定义异步函数 |
| `await` | 等待异步操作完成 |

---

## 3. 用户指南 (User Guide)

### 3.1 最小应用

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

**运行**：
```bash
uvicorn main:app --reload
```

---

### 3.2 路径参数 (Path Parameters)

路径参数是 URL 路径的一部分，用 `{}` 定义。

```python
from fastapi import FastAPI
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

# 基本路径参数
@app.get("/items/{item_id}")
async def read_item(item_id: int):  # 自动类型转换和验证
    return {"item_id": item_id}

# 路径顺序很重要（更具体的路径放前面）
@app.get("/user/me")
async def read_me():
    return {"user_id": "the current user"}

@app.get("/user/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

# 使用枚举
@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}
    return {"model_name": model_name, "message": "Have some residuals"}

# 路径转换器（接收包含 / 的路径）
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
```

**路径参数类型**：
- `int` - 自动验证为整数
- `str` - 字符串
- `float` - 浮点数
- `Enum` - 枚举值
- `:path` - 匹配包含 `/` 的路径

---

### 3.3 查询参数 (Query Parameters)

查询参数是 URL 中 `?` 后面的部分，如 `/items/?skip=0&limit=10`。

```python
from fastapi import FastAPI

app = FastAPI()
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

# 有默认值的查询参数（可选）
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

# 可选查询参数
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}

# 多个查询参数
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id, "short_value": short, "not_short_value": not short}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"description": "This is an amazing item that has a long description"})
    return item

# 多个路径参数和查询参数
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int,
    item_id: str,
    q: str | None = None,
    short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"description": "This is an amazing item that has a long description"})
    return item

# 必需查询参数（没有默认值）
@app.get("/users/{user_id}")
async def read_user(user_id: int, needy: str):
    item = {"user_id": user_id, "needy": needy}
    return item
```

**请求示例**：
```
GET /items/?skip=0&limit=10
GET /items/123?q=search
GET /items/123?short=True
GET /users/42?needy=something
```

---

### 3.4 请求体 (Request Body)

请求体是客户端发送的数据（通常是 JSON），使用 Pydantic 模型定义。

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str                    # 必填
    description: str | None = None  # 可选
    price: float                 # 必填
    tax: float | None = None     # 可选

app = FastAPI()

# POST 请求
@app.post("/items")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

# PUT 请求（路径参数 + 请求体）
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.model_dump()}
```

**请求示例**：
```json
POST /items
{
  "name": "Foo",
  "description": "The foo item",
  "price": 9.99,
  "tax": 1.99
}
```

---

### 3.5 查询参数验证 (Query Validation)

使用 `Query()` 对查询参数添加验证规则。

```python
from typing import Annotated
from fastapi import FastAPI, Query

app = FastAPI()

# 添加验证（推荐使用 Annotated）
@app.get("/items/")
async def read_items(
    q: Annotated[
        str | None,
        Query(
            alias="item_query",      # 别名
            min_length=3,
            max_length=50,
            pattern="^fixedquery$",  # 正则表达式
            title="Query String",
            description="查询字符串"
        )
    ] = None
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

# 查询参数列表（多个值）
@app.get("/items")
async def read_items(q: Annotated[list[str] | None, Query()] = None):
    query_items = {"q": q}
    return query_items

# 请求: /items/?q=foo&q=bar
# 结果: {"q": ["foo", "bar"]}

# 带默认值的列表
@app.get("/items")
async def read_items(q: Annotated[list[str], Query()] = ["foo", "bar"]):
    return {"q": q}
```

**Query 参数**：

| 参数 | 说明 |
|------|------|
| `default` | 默认值 |
| `alias` | 参数别名 |
| `min_length` | 最小长度 |
| `max_length` | 最大长度 |
| `pattern` | 正则表达式 |
| `title` | 文档标题 |
| `description` | 文档描述 |

---

### 3.6 路径参数验证 (Path Validation)

使用 `Path()` 对路径参数添加验证。

```python
from typing import Annotated
from fastapi import FastAPI, Query, Path

app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(
    item_id: Annotated[int, Path(
        title="The ID of item to get",
        ge=0,    # 大于等于 0
        le=1000  # 小于等于 1000
    )],
    q: Annotated[str | None, Query(alias="item_query")] = None,
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results
```

**Path 参数**：

| 参数 | 说明 |
|------|------|
| `ge` | 大于等于 (≥) |
| `gt` | 大于 (>) |
| `le` | 小于等于 (≤) |
| `lt` | 小于 (<) |
| `title` | 文档标题 |
| `description` | 文档描述 |

---

### 3.7 请求体验证 (Request Body Validation)

使用 `Field()` 对请求体字段添加验证。

```python
from typing import Annotated, Literal
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: Annotated[float, Field(ge=0, description="价格必须大于等于0")]
    tax: float | None = Field(None, description="税费")

# 使用 Field 验证
@app.post("/items")
async def create_item(item: Item):
    return item

# 查询参数模型（多个查询参数封装）
class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}  # 禁止额外字段

    limit: int = Field(100, ge=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []

@app.get("/items/")
async def read_items(filter_query: Annotated[FilterParams, Query()]):
    return filter_query
```

---

### 3.8 多个请求体参数

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class User(BaseModel):
    username: str
    full_name: str | None = None

# 多个请求体（会合并为一个 JSON）
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item,
    user: User,
    importance: Annotated[int, Body()]  # 基础类型需要 Body
):
    results = {
        "item_id": item_id,
        "item": item,
        "user": user,
        "importance": importance
    }
    return results
```

**请求示例**：
```json
{
  "item": {
    "name": "Foo",
    "price": 9.99
  },
  "user": {
    "username": "john"
  },
  "importance": 5
}
```

**Body 的作用**：
- 标记参数从 HTTP 请求体读取
- `BaseModel` 类型默认为 Body 参数
- 基础类型（int/str）需要显式使用 `Body()`

---

### 3.9 嵌套模型 (Nested Models)

```python
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()

class Image(BaseModel):
    url: HttpUrl      # 自动验证 URL 格式
    name: str

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = set()   # 自动去重
    image: Image | None = None  # 嵌套模型

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results

# 字典类型的请求体
@app.post("/index-weights")
async def create_index_weights(weights: dict[int, float]):
    return weights
```

---

### 3.10 额外信息 (Extra Info)

给模型添加示例，用于 API 文档。

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "description": "A very long description",
                    "price": 9.15,
                    "tax": 0.1,
                }
            ]
        }
    }

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "item": item}
```

---

### 3.11 额外数据类型 (Extra Data Types)

```python
from fastapi import Body, FastAPI
from typing import Annotated
from uuid import UUID
from datetime import datetime, time, timedelta

app = FastAPI()

@app.put("/items/{item_id}")
async def read_items(
    item_id: UUID,                  # UUID 格式
    start_datetime: Annotated[datetime, Body()],
    end_datetime: Annotated[datetime, Body()],
    process_after: Annotated[timedelta, Body()],
    repeat_at: Annotated[time | None, Body()]
):
    start_process = start_datetime + process_after
    duration = end_datetime - start_datetime
    return {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "process_after": process_after,
        "repeat_at": repeat_at,
        "start_process": start_process,
        "duration": duration
    }
```

**额外类型**：

| 类型 | 说明 | 示例 |
|------|------|------|
| `UUID` | 通用唯一标识符 | `"0d1d2d3d-4d5d-6d7d-8d9d-0d1d2d3d4d5d"` |
| `datetime` | 日期时间 | `"2025-12-31T12:00:00"` |
| `time` | 时间 | `"12:00:00"` |
| `timedelta` | 时间差 | `"PT1H30M"` (1小时30分) |
| `EmailStr` | 邮箱地址 | `"user@example.com"` |

---

### 3.12 响应模型 (Response Model)

定义返回给客户端的数据格式。

```python
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

# 基础响应模型
@app.post("/items/", response_model=Item)
async def create_item(item: Item) -> Any:
    return item

# 列表响应模型
@app.get("/items/", response_model=list[Item])
async def read_items() -> Any:
    return [
        {"name": "Portal Gun", "price": 42.0},
        {"name": "Plumbus", "price": 32.0},
    ]

# 输入/输出模型分离（如密码不返回）
class UserBase(BaseModel):
    name: str
    email: EmailStr
    full_name: str | None = None

class UserIn(UserBase):
    password: str  # 输入包含密码

@app.post("/user/")
async def create_user(user: UserIn) -> UserBase:  # 返回不包含密码
    return user

# 排除默认值（只返回有值的数据）
@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: str):
    return {
        "name": "Foo",
        "price": 50.2,
        "tax": 10.5  # 这个会返回
    }
    # 如果 tax 是默认值，则不会返回

# 不同响应类型
@app.get("/portal")
async def get_portal(teleport: bool = False) -> Response:
    if teleport:
        return RedirectResponse(url="https://www.youtube.com/")
    return JSONResponse(content={"message": "Here's your interdimensional portal."})
```

---

### 3.13 表单和文件 (Form & File)

```python
from fastapi import FastAPI, Form, File, UploadFile
from typing import Annotated

app = FastAPI()

# 表单数据
@app.post("/login/")
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()]
):
    return {"username": username}

# 文件上传
@app.post("/upload/")
async def upload_file(
    file: Annotated[UploadFile, File(description="上传的文件")]
):
    return {
        "文件名": file.filename,
        "文件类型": file.content_type,
        "文件大小": await file.read(),
        "文件对象": file.file
    }
```

| 参数 | 说明 |
|------|------|
| `Form()` | 表单字段（`application/x-www-form-urlencoded`） |
| `File()` | 文件上传（`multipart/form-data`） |
| `UploadFile` | 文件对象，包含 filename、content_type 等 |

---

## 4. 数据库 (Database)

使用 SQLModel 进行数据库操作。

```python
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select

# 定义模型（类似 Java 的实体类）
class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)  # 主键
    name: str = Field(index=True)                            # 索引
    age: int | None = Field(default=None, index=True)
    secret_name: str

# 数据库配置
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

# 创建表
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# 依赖注入：获取 Session
def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI()

# 启动时创建表
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# 创建 Hero
@app.post("/heroes/")
def create_hero(hero: Hero, session: SessionDep) -> Hero:
    session.add(hero)
    session.commit()
    session.refresh(hero)
    return hero

# 查询 Hero 列表
@app.get("/heroes/")
def read_heroes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Hero]:
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes

# 查询单个 Hero
@app.get("/heroes/{hero_id}")
def read_hero(hero_id: int, session: SessionDep) -> Hero:
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero

# 删除 Hero
@app.delete("/heroes/{hero_id}")
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(hero)
    session.commit()
    return {"ok": True}
```

**SQLModel 与 Java 类比**：

| SQLModel | Java (JPA/Hibernate) |
|----------|---------------------|
| `class Hero(SQLModel, table=True)` | `@Entity class Hero` |
| `Field(primary_key=True)` | `@Id` |
| `Field(index=True)` | `@Index` |
| `Session` | `EntityManager` |
| `session.add()` | `entityManager.persist()` |
| `session.commit()` | `transaction.commit()` |
| `select(Hero)` | `SELECT h FROM Hero h` |

---

## 总结

FastAPI 是一个现代、快速的 Web 框架，主要特性：

1. **类型注解** - Python 类型提示提供自动验证和文档
2. **Pydantic** - 数据验证和序列化
3. **异步支持** - 高并发性能
4. **自动文档** - Swagger UI 开箱即用
5. **依赖注入** - `Depends()` 实现优雅的依赖管理
6. **SQLModel** - 简化的数据库操作

**请求参数类型总结**：

| 类型 | 来源 | 装饰器 |
|------|------|--------|
| 路径参数 | URL 路径 | 无 / `Path()` |
| 查询参数 | URL ? 后 | `Query()` |
| 请求体 | JSON Body | `Body()` / BaseModel |
| 表单数据 | Form | `Form()` |
| 文件上传 | multipart | `File()` |

**完整请求示例**：
```
POST /users/123/items/456?q=search&short=true
Headers: Content-Type: application/json
Body:
{
  "item": {"name": "Foo", "price": 9.99},
  "user": {"username": "john"},
  "importance": 5
}
```
