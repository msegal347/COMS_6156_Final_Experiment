@component(base_image='python:3.8-slim')
def squad_data_management(data_path: OutputPath()):
    from squad_downloader import main as download_squad
    download_squad()
    
    from data_preprocessor_squad import process_squad
    process_squad(f'{data_path}/SQuAD_1.1_train.json', f'{data_path}/SQuAD_train_processed.csv')
    
    from data_quality_squad import squad_data_quality_checks
    squad_data_quality_checks(f'{data_path}/SQuAD_train_processed.csv')

    print(f"SQuAD dataset management complete. Processed data at: {data_path}")
