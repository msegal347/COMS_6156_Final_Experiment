# src/kubeflow_pipelines/components/data_preprocessor_squad.py

import json
import pandas as pd
import os

def process_squad(squad_path, processed_path):
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
    
    df = pd.DataFrame(records)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    df.to_csv(processed_path, index=False)
    print(f"Processed SQuAD dataset saved to {processed_path}")

# Example usage
if __name__ == "__main__":
    process_squad('./data/raw/SQuAD_1.1_train.json', './data/processed/SQuAD_1.1_train_processed.csv')
    process_squad('./data/raw/SQuAD_1.1_dev.json', './data/processed/SQuAD_1.1_dev_processed.csv')    
    process_squad('./data/raw/SQuAD_2.0_train.json', './data/processed/SQuAD_2.0_train_processed.csv')
    process_squad('./data/raw/SQuAD_2.0_dev.json', './data/processed/SQuAD_2.0_dev_processed.csv')