from shutil import copyfileobj

import filetype
import fitz
import pytesseract
from fastapi import FastAPI, Response, UploadFile, status
from PIL import Image

from ml import bert_model
from models import QARequest

app = FastAPI()


@app.get("/")
async def root():
    return {"answer": "Hello World"}


@app.post("/qa")
async def qa(req: QARequest):
    """
    Blocking call due to underlying BERT Model
    """
    # answer = bert_model.run_bert_model(question=req.text, paragraph=req.text_pair)
    answer = bert_model.oracle(question=req.text, context=req.text_pair)
    return {"answer": answer}


@app.post(path="/uploadfile/", status_code=status.HTTP_200_OK)
async def create_upload_file(file: UploadFile, response: Response):
    """
    Blocking call due to IO
    """
    context = ""

    if file.size > 1000000:  # ~1 MB
        response.status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    else:
        kind = filetype.guess(file.file)

        match kind.MIME:
            case "application/pdf":

                with open(file.filename, "wb") as fdst:
                    copyfileobj(file.file, fdst)

                with fitz.open(file.filename) as document:
                    for page in document:
                        context = page.get_text()

            case "image/jpeg":
                context = pytesseract.image_to_string(Image.open(file.file))

            case _:
                response.status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

        return {"context": context.replace(r"\n", "").strip().encode()}
