import pandas as pd
import logging
import os

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def squad_data_quality_checks(file_path):
    if not os.path.exists(file_path):
        logging.error(f"The file {file_path} does not exist.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to read the file {file_path}. Error: {e}")
        return
    
    if df.empty:
        logging.warning("The DataFrame is empty. No data to check.")
        return
    
    # Check for missing values
    if df.isnull().any().any():
        logging.error("Missing values found in SQuAD dataset.")
        return
    
    # Check for uniqueness of 'id'
    if not df['id'].is_unique:
        logging.error("'id' column in SQuAD dataset is not unique.")
        return
    
    # Optional: Check for non-empty context and questions
    if (df['context'].str.strip() == '').any() or (df['question'].str.strip() == '').any():
        logging.error("Empty strings found in 'context' or 'question' columns.")
        return

    logging.info("SQuAD data quality checks passed.")

# Example usage
if __name__ == "__main__":
    process_squad_1_train_csv = './data/processed/SQuAD_1.1_train_processed.csv'
    process_squad_1_dev_csv = './data/processed/SQuAD_1.1_dev_processed.csv'
    process_squad_2_train_csv = './data/processed/SQuAD_2.0_train_processed.csv'
    process_squad_2_dev_csv = './data/processed/SQuAD_2.0_dev_processed.csv'
    squad_data_quality_checks(process_squad_1_train_csv)
    squad_data_quality_checks(process_squad_1_dev_csv)
    squad_data_quality_checks(process_squad_2_train_csv)
    squad_data_quality_checks(process_squad_2_dev_csv)
