@component(base_image='python:3.8-slim')
def imagenet_data_management(data_path: OutputPath()):
    from ImageNet_downloader import main as download_imagenet
    download_imagenet()
    
    from data_preprocessor_imagenet import process_imagenet
    process_imagenet(f'{data_path}/tiny_imagenet', f'{data_path}/tiny_imagenet_processed')
    
    # Assuming a metadata CSV is generated/available for quality checks
    from data_quality_imagenet import imagenet_data_quality_checks
    imagenet_data_quality_checks(f'{data_path}/imagenet_metadata.csv')

    print(f"ImageNet dataset management complete. Processed data at: {data_path}")
