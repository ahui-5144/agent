from typing import Annotated

from fastapi import FastAPI, Query, Path

app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(
        item_id: Annotated[int, Path(title="The ID of item to get")],
        q: Annotated[str | None, Query(alias="item_query")] = None,
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results