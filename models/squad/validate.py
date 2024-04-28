from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset, load_metric
from collections import defaultdict
import numpy as np
import torch

### Adapted from: https://github.com/kamalkraj/BERT-SQuAD/tree/master

def prepare_validation_features(examples, tokenizer):
    # Tokenization
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if tokenized_examples["token_type_ids"][i][0] == 0 else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def postprocess_qa_predictions(examples, features, predictions, version_2_with_negative=False):
    assert len(predictions["start_logits"]) == len(predictions["end_logits"]) == len(features)
    all_start_logits, all_end_logits = predictions["start_logits"], predictions["end_logits"]

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The final predictions per example.
    all_predictions = defaultdict(str)
    for example_id, feature_indices in features_per_example.items():
        min_null_score = None  # Only used if version_2_with_negative is True
        context = examples[example_id_to_index[example_id]]["context"]
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for this feature.
            start_indexes = np.argsort(start_logits)[-1 : -11 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -11 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Ignore out-of-scope answers.
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > 30:
                        continue

                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[offset_mapping[start_index][0]:offset_mapping[end_index][1]]
                        }
                    )

        if version_2_with_negative:
            # Compare and save the null score with the score of best non-null answer.
            if min_null_score < max([va["score"] for va in valid_answers], default=0):
                all_predictions[example_id] = ""
            else:
                best_answer = max(valid_answers, key=lambda x: x["score"], default={"text": ""})
                all_predictions[example_id] = best_answer["text"]
        else:
            best_answer = max(valid_answers, key=lambda x: x["score"], default={"text": ""})
            all_predictions[example_id] = best_answer["text"]

    return all_predictions

def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(examples=examples, features=features, predictions=predictions, version_2_with_negative=True)
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def compute_metrics(p, metric):
    # Pass the metric explicitly
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def main():
    model_path = './path/to/your/saved/model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)

    # Load validation data
    datasets = load_dataset("squad_v2")
    validation_features = datasets['validation'].map(
        lambda examples: prepare_validation_features(examples, tokenizer),
        batched=True,
        remove_columns=datasets["validation"].column_names
    )

    # Define trainer
    args = TrainingArguments(
        per_device_eval_batch_size=8,
        dataloader_num_workers=2,
        output_dir="./results",
    )
    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        eval_dataset=validation_features,
        tokenizer=tokenizer,
    )

    # Load metric
    metric = load_metric("squad_v2")

    # Run validation
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()
