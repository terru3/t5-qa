# Imports
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from constants import *
from utils import set_seed, device

set_seed()

# Data

squad = load_dataset("squad", split="train")
squad = squad.shuffle(seed=42)
squad = squad.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)

# Preprocessing

#### Goal: Tokenize question+context, find start and end token positions of answer, delete unnecessary features
# Instead of character indices, we want the token indices where the answer starts and ends (e.g. (38, 41) => answer string is the 38th to 41th tokens (inclusive))

def tokenize_preprocess(examples):
    """
    This function tokenizes question-context pairs and obtains the start and end token indices of the answers.
    It is intended for preprocessing use on the SQuAD v1.1 dataset.

    Input:
        -examples: Huggingface Dataset instance with keys ['question', 'context', 'answers'], etc., of lists
    Returns:
        -The same Dataset instance with keys replaced by tokenized
            'input_ids', 'attention_mask', 'start_positions', and 'end_positions'
                For each data instance, we have:
                -Input_ids:  List of integer token id's, used by the tokenizer to represent the given string
                -Attention_mask: List of [1,1,..0], where a 1 indicates an underlying character and 0 indicates padding
                -Start_positions: Index indicating the token position in `input_ids` where the answer begins
                -End_positions: Index indicating the token position in `input_ids` where the answer ends
    """
    # remove unnecessary whitespace in questions
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=SEQ_LENGTH,
        truncation="only_second", # only truncate the context not the question.
        padding="max_length",
    )

    answers = examples["answers"]
    start_token_list = []
    end_token_list = []

    for i in range(len(answers)):

        # obtain end character indices
        answer = answers[i]
        answer_str = answer['text'][0]

        start_idx = answer['answer_start'][0]
        end_idx = start_idx + len(answer_str)

        # Rather than raw start and end character indices, we need the start and end TOKEN indices
        # ——e.g. answer starts at token 32, ends at token 39

        # We use char_to_token with sequence_index=1, so that we look
        # only at the second string (the context paragraph)
        start_token = inputs.char_to_token(i, start_idx, sequence_index=1)
        end_token = inputs.char_to_token(i, end_idx-1, sequence_index=1)

        # if char_to_token outputs None, then that token was actually truncated by tokenization earlier
        # so we set to max token length
        if start_token is None:
            start_token = SEQ_LENGTH
        if end_token is None:
            end_token = SEQ_LENGTH

        start_token_list.append(start_token)
        end_token_list.append(end_token)

    # append sequence_ids from tokenization, necessary for post-processing
    map_batch_size = len(inputs['input_ids'])
    seq_ids = [inputs.sequence_ids(i) for i in range(map_batch_size)]

    inputs["start_positions"] = start_token_list
    inputs["end_positions"] = end_token_list
    inputs["sequence_ids"] = seq_ids

    return inputs

# Map preprocessing function to the dataset (approx 2.5 minutes)
tokenized_squad = squad.map(tokenize_preprocess,
                                batched=True,
                                remove_columns=squad["train"].column_names)

# Format dataset to use tensors rather than native Python lists
tokenized_squad = tokenized_squad.with_format('torch')

"""## Dataloading"""

train_dataloader = DataLoader(tokenized_squad['train'],
                              shuffle=True,
                              batch_size=BATCH_SIZE)

val_dataloader = DataLoader(tokenized_squad['test'],
                            shuffle=True,
                            batch_size=BATCH_SIZE)