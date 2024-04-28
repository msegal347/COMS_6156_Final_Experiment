import argparse
import logging
import os
import random
import glob
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import (WEIGHTS_NAME, BertConfig, BertForQuestionAnswering, BertTokenizer,
                          XLMConfig, XLMForQuestionAnswering, XLMTokenizer, XLNetConfig,
                          XLNetForQuestionAnswering, XLNetTokenizer, get_scheduler)

# Import necessary components for mixed precision training
from torch.cuda.amp import GradScaler, autocast

from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions, RawResultExtended, write_predictions_extended)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

base_image = 'gcr.io/coms-6156-kubeflow/squad:latest'

# Define the component to perform evaluation
@component(base_image=base_image)
def evaluate_model(
    model_dir: InputPath,
    data_dir: InputPath,
    result_path: OutputPath(str),
    model_type: str = 'bert',
    max_seq_length: int = 384,
    batch_size: int = 8,
    device: str = 'cuda',
    seed: int = 42,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    verbose_logging: bool = True,
    version_2_with_negative: bool = False
):
    import numpy as np
    from utils_squad import load_and_cache_examples, to_list
    from utils_squad_evaluate import evaluate_on_squad, EVAL_OPTS

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Define tokenizer and model based on model_type
    MODEL_CLASSES = {
        'bert': (BertForQuestionAnswering, BertTokenizer),
        'xlnet': (XLNetForQuestionAnswering, XLNetTokenizer),
        'xlm': (XLMForQuestionAnswering, XLMTokenizer),
    }
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    model = model_class.from_pretrained(model_dir)
    model.to(device)

    # Load and prepare data
    dataset, examples, features = load_and_cache_examples(data_dir, tokenizer, max_seq_length=max_seq_length, evaluate=True, output_examples=True)

    # Prepare DataLoader
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    model.eval()
    all_results = []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }
            outputs = model(**inputs)
            for i in range(outputs[0].shape[0]):
                result = {
                    'unique_id': int(features[i].unique_id),
                    'start_logits': to_list(outputs[0][i]),
                    'end_logits': to_list(outputs[1][i])
                }
                all_results.append(result)

    # Write predictions and evaluate
    output_prediction_file = os.path.join(result_path, "predictions_.json")
    output_nbest_file = os.path.join(result_path, "nbest_predictions_.json")
    null_odds_file = os.path.join(result_path, "null_odds_.json") if version_2_with_negative else None

    write_predictions(
        examples, 
        features, 
        all_results, 
        n_best_size, 
        max_answer_length, 
        do_lower_case=False, 
        output_prediction_file=output_prediction_file,
        output_nbest_file=output_nbest_file,
        output_null_log_odds_file=null_odds_file,
        verbose_logging=verbose_logging,
        version_2_with_negative=version_2_with_negative,
        null_score_diff_threshold=0.0
    )

    # Evaluate using the SQuAD script
    evaluate_options = EVAL_OPTS(data_file=data_dir, pred_file=output_prediction_file, na_prob_file=null_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print("Evaluation results:", results)

    # Optionally save the results
    with open(result_path, 'w') as result_file:
        result_file.write(str(results))
