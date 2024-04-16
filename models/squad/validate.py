from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np
import torch

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

def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(examples=examples, features=features, predictions=predictions, version_2_with_negative=True)
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def compute_metrics(p: EvalPrediction):
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
