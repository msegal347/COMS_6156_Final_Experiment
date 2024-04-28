import kfp
from kfp.v2.dsl import component, InputPath, OutputPath

base_image = 'gcr.io/coms-6156-kubeflow/squad:latest'

@component(base_image=base_image)
def process_squad_data(squad_path: InputPath(), processed_path: OutputPath()):
    import json
    import pandas as pd
    import os

    # Load the SQuAD data from the specified input path
    with open(squad_path, 'r') as f:
        squad_data = json.load(f)
    
    # Flatten the dataset to a pandas DataFrame
    records = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                record = {
                    'context': paragraph['context'],
                    'question': qa['question'],
                    'id': qa['id']
                }
                if 'answers' in qa:
                    record['answers'] = [a['text'] for a in qa['answers']]
                else:
                    record['answers'] = []
                records.append(record)
    
    # Convert the records into a DataFrame
    df = pd.DataFrame(records)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    # Save the processed data to the specified output path
    df.to_csv(processed_path, index=False)
    print(f"Processed SQuAD dataset saved to {processed_path}")
