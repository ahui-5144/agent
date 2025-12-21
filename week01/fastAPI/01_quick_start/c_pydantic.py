from datetime import datetime
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str = "john wick"
    signup_ts: datetime | None = None
    friends: list[int] = []

external_data = {
    "id": "123",
    "signup_ts": datetime.now(),
    "friends": [1, "2", b"3"],
}

user = User(**external_data)
print(user)
# > id=123 name='john wick' signup_ts=datetime.datetime(2025, 12, 16, 16, 16, 51, 539556) friends=[1, 2, 3]
print(user.id)