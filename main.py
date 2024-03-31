from fastapi import FastAPI, File, UploadFile
from shutil import copyfileobj

from models import QARequest
from ml import bert_model

from typing import Annotated

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/qa")
async def qa(req: QARequest):
    """
    Blocking call due to underlying BERT Model
    """
    answer = bert_model.run_bert_model(question=req.text, paragraph=req.text_pair)
    return {"answer": answer}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    """
    Blocking call due to IO
    """
    # explore shutil.copyfileobj() and asyncio/aiofiles for memory-efficiency and non-blocking nature
    # contents = await file.read()
    # with open(file.filename, "wb") as buffer:
    #    # buffer.write(contents)
    copyfileobj(file.file, open(file.filename, "wb"))
    return {"name": file.filename, "ext": file.content_type}
