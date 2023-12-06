import random
import string
from functools import partial

import numpy as np
import torch
from transformers import AutoTokenizer


def set_seed(seed=42):
    """
    Sets seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = (
        True  # only applies to CUDA convolution operations
    )
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)


def retrieve_preds_and_labels(
    start_logits,
    end_logits,
    input_ids,
    seq_ids,
    start_pos=None,
    end_pos=None,
    n_best=10,
    max_answer_len=30,
    inference=False,
):
    """
    Mapping helper function which post-processes and decodes fine-tuned T5ForQuestionAnswering model outputs
    as well as decoded ground truth labels.

    Inputs:
      -Start_logits and end_logits refer to the model output
      -Input_ids and seq_ids refer to the tokenized input
      -All are tensors of length `seq_len`
      -Start_pos and end_pos refer to the token indices of the ground truth labels if given and are tensors of length 1
      -n_best (int): Number of each of start and end indices to consider as candidates (no need to check all `seq_len` logits)
      -max_answer_len (int): Max token length of a predicted answer
      -inference (bool=True): If true, processes labels
    Returns:
      -Tuple of two lists, each of length batch_size
        -Decoded model predictions
        -Decoded ground truth labels
      -If inference=True, only the predictions are returned as a 1-tuple

    """
    assert (
        isinstance(n_best, int)
        and isinstance(max_answer_len, int)
        and n_best > 0
        and max_answer_len > 0
    )

    start_idx_list = np.argsort(start_logits.cpu().numpy())[-1 : (-n_best - 1) : -1]
    end_idx_list = np.argsort(end_logits.cpu().numpy())[-1 : (-n_best - 1) : -1]
    # requires cpu to use np.argsort and requires numpy for negative indexing
    # start_idx_list, end_idx_list are lists of length `n_best`. Now we check all n_best^2 combinations

    valid_answers = []
    for start_idx in start_idx_list:
        for end_idx in end_idx_list:
            # Ignore out-of-scope answers (i.e. indices of predicted answer is outside the context)
            if (
                # to do so, we make use of sequence id. If not 1 then the index is outside the context, because 0 => part of qusstion
                # and None => special token
                seq_ids[start_idx].item() != 1
                or seq_ids[end_idx].item() != 1
            ):
                continue
            # Ignore answers with negative length or > max_answer_len
            if start_idx > end_idx or end_idx - start_idx + 1 > max_answer_len:
                continue

            # If this start end index pair survives it's valid
            # we sum the start and end logits
            valid_answers.append(
                {
                    "score": start_logits[start_idx] + end_logits[end_idx],
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

    # take prediction with max score, only decode this prediction (no need to decode all candidates)
    final_preds = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    final_decoded_preds = tokenizer.decode(
        input_ids[final_preds["start_idx"] : (final_preds["end_idx"] + 1)]
    )

    # Now decode ground truth labels
    if not inference:
        labels = tokenizer.decode(input_ids[start_pos : (end_pos + 1)])
        return final_decoded_preds, labels
    else:
        return (final_decoded_preds,)


def postprocess(batch, output, inference=False, **kwargs):
    """
    Postprocesses and decodes model output (and ground truth labels if any).

    Inputs;
      -batch: The data batch returned from the DataLoader
      -output: Output of the model when given `batch`
      -inference (bool=False): Indicates if labels are available and decodes + returns them if so
    Returns:
      -2-tuple of numpy arrays of length `batch_size` indicating the model predictions and the ground truth labels respectively
      -Note: If set to inference mode (i.e. no labels), only the predictions are returned, and not in 1-tuple form.
    """

    # batch size used
    b_size = batch["input_ids"].size(0)

    # prepare map function with fixed inference and keyword arguments
    mapfunc = partial(retrieve_preds_and_labels, inference=inference, **kwargs)

    # if inference, no start/end positions, and we initialize placeholder tensors
    if inference:
        start_pos, end_pos = torch.empty((b_size, 1)), torch.empty((b_size, 1))
    else:
        start_pos, end_pos = batch["start_positions"], batch["end_positions"]

    # map helper function
    postprocessed_output = list(
        map(
            mapfunc,
            output.start_logits,
            output.end_logits,
            batch["input_ids"],
            batch["sequence_ids"],
            start_pos,
            end_pos,
        )
    )

    # output shape above: list of length `batch_size` of 2-tuples (pred, label) or 1-tuple (pred, )

    preds = np.array([postprocessed_output[i][0] for i in range(b_size)])
    if not inference:
        labels = np.array([postprocessed_output[i][1] for i in range(b_size)])
        return preds, labels
    else:
        return preds


def normalization(text):
    """
    Normalizes a given text by fixing whitespaces, converting to lowercase, and removing punctuation.
    This function does not remove stopwords, articles, or translate numbers to words as these actions
    can affect the length of the strings and thus the F-1 score.

    Input:
        -text (str): Text string to be normalized
    Returns:
        -The normalized text string
    """
    # Fix whitespaces, convert lowercase
    text = " ".join(text.split()).lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text
