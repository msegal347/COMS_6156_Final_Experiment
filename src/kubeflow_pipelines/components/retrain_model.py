import kfp
from kfp.v2.dsl import component, InputPath, OutputPath


@component
def retrain_model(
    data_dir: InputPath,
    model_dir: OutputPath,
    hyperparameters: InputPath
):
    """Retrain the model using the best hyperparameters found by Katib."""
    import json
    import torch
    from torch.utils.data import DataLoader
    from transformers import AdamW, get_scheduler
    from utils_squad import load_and_cache_examples

    # Load hyperparameters
    with open(hyperparameters, 'r') as f:
        params = json.load(f)
    
    learning_rate = float(params.get("learning_rate", 5e-5))
    batch_size = int(params.get("batch_size", 16))

    # Load the dataset
    dataset = load_and_cache_examples(data_dir, evaluate=False)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Load model
    model = torch.load(model_dir) 
    model.train()

    # Training setup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))

    # Training loop
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Save the retrained model
    torch.save(model, model_dir)
    print("Model retrained and saved at", model_dir)
