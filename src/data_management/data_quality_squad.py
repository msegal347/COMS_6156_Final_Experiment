import pandas as pd

def squad_data_quality_checks(file_path):
    df = pd.read_csv(file_path)
    
    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("Missing values found in SQuAD dataset.")
    
    # Check for uniqueness of 'id'
    if not df['id'].is_unique:
        raise ValueError("'id' column in SQuAD dataset is not unique.")
    
    print("SQuAD data quality checks passed.")

# Example usage
if __name__ == "__main__":
    process_squad_csv = './data/processed/SQuAD_train_processed.csv'
    squad_data_quality_checks(process_squad_csv)
