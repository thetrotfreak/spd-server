import pathlib

import torch
from transformers import BertForQuestionAnswering, BertTokenizer


def run_bert_model(question, paragraph):
    # model = BertForQuestionAnswering.from_pretrained(
    #     "bert-large-uncased-whole-word-masking-finetuned-squad"
    # )

    # tokenizer = BertTokenizer.from_pretrained(
    #     "bert-large-uncased-whole-word-masking-finetuned-squad"
    # )
    model = BertForQuestionAnswering.from_pretrained(
        pathlib.Path(
            "ml/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
        ).absolute(),
        local_files_only=True,
        # from_flax=True,
    )

    tokenizer = BertTokenizer.from_pretrained(
        pathlib.Path(
            "ml/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
        ).absolute(),
        local_files_only=True,
        # from_flax=True,
    )

    # answer_text is paragraph
    input_ids = tokenizer.encode(question, paragraph)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # Run our example through the model.
    outputs = model(
        torch.tensor([input_ids]),  # The tokens representing our input text.
        token_type_ids=torch.tensor(
            [segment_ids]
        ),  # The segment IDs to differentiate question from answer_text
        return_dict=True,
    )

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Combine the tokens in the answer and print it out.
    answer = " ".join(tokens[answer_start : answer_end + 1])
    return answer
