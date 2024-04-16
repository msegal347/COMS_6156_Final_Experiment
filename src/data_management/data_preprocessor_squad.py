# src/kubeflow_pipelines/components/data_preprocessor_squad.py

import json
import pandas as pd

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
    df.to_csv(processed_path, index=False)
    print(f"Processed SQuAD dataset saved to {processed_path}")

# Example usage
if __name__ == "__main__":
    process_squad('./data/raw/SQuAD_1.1_train.json', './data/processed/SQuAD_train_processed.csv')
