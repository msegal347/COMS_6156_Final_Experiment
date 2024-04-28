import kfp
from kfp.v2.dsl import component, InputPath, OutputPath

base_image = 'gcr.io/coms-6156-kubeflow/squad:latest'

@component
def train_model(
    model_dir: InputPath,
    data_dir: InputPath,
    output_dir: OutputPath,
    train_file: str,
    model_type: str = 'bert',
    per_gpu_train_batch_size: int = 8,
    num_train_epochs: int = 3,
    max_steps: int = -1,
    learning_rate: float = 5e-5,
    adam_epsilon: float = 1e-8,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    logging_steps: int = 50,
    save_steps: int = 50,
    seed: int = 42,
    fp16: bool = False,
):
    """Train a model using PyTorch and the Transformers library."""
    import os
    import torch
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset
    from transformers import get_scheduler, AdamW
    from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering
    from torch.cuda.amp import GradScaler, autocast
    from utils_squad import load_and_cache_examples
    from tqdm import tqdm, trange

    # Setup device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load tokenizer and model
    config_class, model_class, tokenizer_class = {
        'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer)
    }[model_type]
    config = config_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    model = model_class.from_pretrained(model_dir, config=config)
    model.to(device)

    # Prepare training dataset
    train_dataset = load_and_cache_examples(train_file, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=per_gpu_train_batch_size)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    # Setup training
    model.zero_grad()
    scaler = GradScaler() if fp16 else None
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=fp16):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()

            if step % save_steps == 0:
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

    return output_dir