from pydantic import BaseModel


class QARequest(BaseModel):
    text: str
    text_pair: str
