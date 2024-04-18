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

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def create_model_and_tokenizer(args, model_classes):
    config_class, model_class, tokenizer_class = model_classes[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
    return model, tokenizer, model_class

def cleanup_checkpoints(directory, max_keep=3):
    """Keep only the `max_keep` most recent checkpoint directories."""
    checkpoints = sorted([os.path.join(directory, d) for d in os.listdir(directory) if d.startswith('checkpoint-')],
                         key=os.path.getmtime, reverse=True)
    if len(checkpoints) > max_keep:
        for old_checkpoint in checkpoints[max_keep:]:
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Calculate total steps
    t_total = args.max_steps if args.max_steps > 0 else len(train_dataloader) * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    scaler = GradScaler() if args.fp16 else None

    model.zero_grad()
    global_step = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2] if args.model_type != 'xlm' else None, 'start_positions': batch[3], 'end_positions': batch[4]}

            if args.fp16:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()  # Update the learning rate.
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and global_step % args.logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)  # Change get_last_lr() to get_lr() depending on PyTorch version
                tb_writer.add_scalar('loss', loss.item(), global_step)

            if global_step % args.save_steps == 0 or global_step == t_total:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)

                # Call cleanup_checkpoints to remove old checkpoints
                cleanup_checkpoints(args.output_dir, max_keep=3)

            if global_step == t_total:
                break
    tb_writer.close()
    return global_step, loss.item()



def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.predict_file,
                        model.config.start_n_top, model.config.end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--train_file", type=str, required=True, help="SQuAD json for training.")
    parser.add_argument("--predict_file", type=str, required=True, help="SQuAD json for predictions.")
    parser.add_argument("--model_type", type=str, required=True, choices=['bert', 'xlnet', 'xlm'], help="Model type selected.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")
    # Other parameters
    parser.add_argument("--config_name", type=str, default="", help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default="", help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", type=str, default="", help="Where to store the pre-trained models downloaded from s3")
    parser.add_argument('--version_2_with_negative', action='store_true', help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0, help="Threshold for predicting null.")
    parser.add_argument("--max_seq_length", type=int, default=384, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--doc_stride", type=int, default=128, help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", type=int, default=64, help="The maximum number of tokens for the question.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=8, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=8, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", type=int, default=20, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", type=int, default=30, help="The maximum length of an answer that can be generated.")
    parser.add_argument("--verbose_logging", action='store_true', help="If true, all of the warnings related to data processing will be printed.")
    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true', help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    setup_logging()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if args.server_ip and args.server_port:
        import debugpy
        print("Waiting for debugger attach")
        debugpy.listen((args.server_ip, args.server_port))
        debugpy.wait_for_client()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 1

    args.device = device

    set_seed(args)
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    }
    model, tokenizer, model_class = create_model_and_tokenizer(args, MODEL_CLASSES)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)
        logger.info("Results: {}".format(results))

if __name__ == "__main__":
    main()