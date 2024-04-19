from transformers import pipeline


def oracle(question: str, context: str):
    pipe = pipeline(
        "question-answering",
        model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    )
    result: dict = pipe(question=question, context=context)
    return result.get("answer")
