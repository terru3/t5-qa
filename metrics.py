import numpy as np

from utils import normalization

# Evaluation functions

def exact_match(preds, labels):
  """
  Calculates the exact match score between predictions and labels.
  Normalizes the predictions and labels first, then computes the proportion of equality between normalized predictions and labels.

  Input:
      -preds (np.array): Array of prediction strings
      -labels (np.array): Array of label strings
  Returns:
      -Exact match score (float) between the normalized predictions and labels

  """
  # Normalize predictions and labels
  preds = np.vectorize(normalization)(preds)
  labels = np.vectorize(normalization)(labels)

  return np.mean(preds == labels)

def f1(preds, labels):
  """
  Computes F-1 score word-level.

  Input:
      -preds (np.array): Array of prediction strings
      -labels (np.array): Array of label strings
  Returns:
      -Mean F-1 score (float) for all pairs of normalized predictions and labels
  """

  f1_list = []

  # Normalize predictions and labels
  preds = np.vectorize(normalization)(preds)
  labels = np.vectorize(normalization)(labels)

  # Calculates F-1 Score for each pair of preds & labels
  for i in range(len(preds)):
    pred_tokens = preds[i].split()
    act_tokens = labels[i].split()

    common_tokens = set(pred_tokens) & set(act_tokens)
    if len(common_tokens) == 0:
      f1_list.append(0)
    else:
      pre = len(common_tokens) / len(pred_tokens)
      rec = len(common_tokens) / len(act_tokens)
      f1 =  2 * (pre * rec) / (pre + rec)
      f1_list.append(f1)

  return np.mean(f1_list)

def sas(cross_encoder, preds, labels):
  """
  Computes Semantic Answer Similarity (SAS) scores between predictions and labels via a cross-encoder.

  Input:
      -cross_encoder: Cross-encoder model used for prediction
      -preds (np.array): Array of prediction strings
      -labels (np.array): Array of label strings
  Returns:
      -Mean SAS score (float) for all prediction-label pairs

  """
  cross_encoder_input = [(preds[i], labels[i]) for i in range(len(preds))]
  sas_scores = cross_encoder.predict(cross_encoder_input)

  return sas_scores.mean()

