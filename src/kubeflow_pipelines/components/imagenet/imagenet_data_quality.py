import kfp
from kfp.v2.dsl import component, InputPath

base_image = 'gcr.io/coms-6156-kubeflow/imagenet:latest'

@component(base_image=base_image)
def imagenet_data_quality_check(metadata_file_path: InputPath()):
    import pandas as pd
    import logging
    import os

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(metadata_file_path):
        logging.error(f"The file {metadata_file_path} does not exist.")
        return

    try:
        df = pd.read_csv(metadata_file_path)
    except Exception as e:
        logging.error(f"Failed to read the file {metadata_file_path}. Error: {e}")
        return

    if df.empty:
        logging.warning("The DataFrame is empty. No data to check.")
        return

    # Check for missing values
    if df.isnull().any().any():
        logging.error("Missing values found in ImageNet metadata.")
        return

    min_images_per_class = 500  
    class_distribution = df['class_id'].value_counts()
    if (class_distribution < min_images_per_class).any():
        logging.error(f"Some classes have fewer than {min_images_per_class} images.")
        return

    logging.info("ImageNet data quality checks passed.")
