import pandas as pd

def imagenet_data_quality_checks(metadata_file_path):
    """
    Perform data quality checks on the ImageNet metadata.
    """
    df = pd.read_csv(metadata_file_path)
    
    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("Missing values found in ImageNet metadata.")
    
    min_images_per_class = 500  
    class_distribution = df['class_id'].value_counts()
    if (class_distribution < min_images_per_class).any():
        raise ValueError(f"Some classes have fewer than {min_images_per_class} images.")
    
    print("ImageNet data quality checks passed.")

if __name__ == "__main__":
    imagenet_metadata_csv = './data/processed/imagenet_metadata.csv'
    imagenet_data_quality_checks(imagenet_metadata_csv)
