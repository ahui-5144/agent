from fastapi import Body, FastAPI
from typing import Annotated
from uuid import UUID
from datetime import datetime, time, timedelta

from watchfiles.run import start_process

app = FastAPI()

@app.put("/items/{item_id}")
async def read_items(
        item_id: UUID,
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