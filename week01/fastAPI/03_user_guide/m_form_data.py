from fastapi import FastAPI, Form, File, UploadFile
from typing import Annotated

app = FastAPI()

@app.post("/login/")
async def login(username: Annotated[str, Form()], password: Annotated[str, Form()]):
    return {"username": username}

@app.post("/upload/")
async def upload_file(
    # Annotated：绑定「文件类型」和「文件来源元数据」
    file: Annotated[UploadFile, File(description="上传的文件")]
):
    # UploadFile 提供的核心属性/方法：
    return {
        "文件名": file.filename,          # 文件名（如 "test.png"）
        "文件类型": file.content_type,    # 文件MIME类型（如 "image/png"）
        "文件大小": await file.read(),    # 读取文件内容（bytes 类型）
        "文件对象": file.file             # 原始文件对象（可直接写入磁盘）
    }