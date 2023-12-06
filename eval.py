# Imports
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from metrics import exact_match, f1, sas
from utils import set_seed, device, postprocess

# Evaluation
def eval_acc(model, val_dataloader):
  """
  Evaluates the accuracy of a model on the validation dataset using various metrics:
    -EM, F1, METEOR, ROUGE-1, ROUGE-2, ROUGE-L, and SAS

  Input:
      -model: Model to be evaluated
      -val_dataloader: Data loader for the validation dataset
  Returns:
      -A dictionary containing the mean accuracy scores for each metric
  """

  # Setting up the evaluation metrics
  meteor = evaluate.load('meteor')
  rouge = evaluate.load('rouge')
  bleu = evaluate.load("bleu")
  cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')

  # Put the model into evaluation mode
  model.eval()

  # Initialize lists to store accuracy scores
  em_list = []
  f1_list = []
  meteor_list = []
  rouge_1_list, rouge_2_list, rouge_L_list = [], [], []
  bleu_list = []
  sas_list = []

  # Evaluate model
  with torch.no_grad():
    print('Evaluating Validation Accuracies:')
    for batch in tqdm(val_dataloader):
      output = model(batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    start_positions=batch['start_positions'].to(device),
                    end_positions=batch['end_positions'].to(device))

      preds, labels = postprocess(batch, output, inference=False,
                                  n_best=10, max_answer_len=30)

      # Compute accuracy
      em_val = exact_match(preds, labels)
      f1_val = f1(preds, labels)
      meteor_val = meteor.compute(predictions=preds, references=labels)['meteor']

      rouge_val = rouge.compute(predictions=preds, references=labels)
      rouge_1 = rouge_val['rouge1']
      rouge_2 = rouge_val['rouge2']
      rouge_L = rouge_val['rougeL']

      bleu_val = bleu.compute(predictions=preds, references=labels)['bleu']
      sas_val = sas(cross_encoder, preds, labels)

      # Append accuracy scores to the corresponding lists
      em_list.append(em_val)
      f1_list.append(f1_val)
      meteor_list.append(meteor_val)
      rouge_1_list.append(rouge_1)
      rouge_2_list.append(rouge_2)
      rouge_L_list.append(rouge_L)
      bleu_list.append(bleu_val)
      sas_list.append(sas_val)

  # Compute and print average accuracy scores
  em_score = np.mean(em_list)
  f1_score = np.mean(f1_list)
  meteor_score = np.mean(meteor_list)
  rouge_1_score = np.mean(rouge_1_list)
  rouge_2_score = np.mean(rouge_2_list)
  rouge_L_score = np.mean(rouge_L_list)
  bleu_score = np.mean(bleu_list)
  sas_score = np.mean(sas_list)

  print(f"\n\nExact Match: {em_score}")
  print(f"F1: {f1_score}")
  print(f"METEOR: {meteor_score}")
  print(f"ROUGE-1: {rouge_1_score}")
  print(f"ROUGE-2: {rouge_2_score}")
  print(f"ROUGE-L: {rouge_L_score}")
  print(f"BLEU: {bleu_score}")
  print(f"SAS: {sas_score}")

  metrics_dict = {'EM': em_score,
                 'F1': f1_score,
                 'METEOR': meteor_score,
                 'ROUGE-1': rouge_1_score,
                 'ROUGE-2': rouge_2_score,
                 'ROUGE-L': rouge_L_score,
                 'BLEU': bleu_score,
                 'SAS': sas_score}

  return metrics_dict