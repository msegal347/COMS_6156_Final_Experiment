import kfp
from kfp import dsl
from kubernetes.client.models import V1EnvFromSource, V1SecretKeySelector
from kubeflow_pipelines.components.squad.squad_downloader import download_squad_data
from kubeflow_pipelines.components.squad.squad_data_preprocessor import process_squad_data
from kubeflow_pipelines.components.squad.squad_data_quality import squad_data_quality_check
from kubeflow_pipelines.components.squad.squad_model_train import train_model
from kubeflow_pipelines.components.squad.squad_model_evaluation import evaluate_model
from kubeflow_pipelines.components.katib_launcher import katib_launcher
from kubeflow_pipelines.components.hyperparamter_retrieval import fetch_best_hyperparameters
from kubeflow_pipelines.components.retrain_model import retrain_model

@dsl.pipeline(
    name='squad-training-pipeline',
    description='A comprehensive pipeline that handles SQuAD dataset from download to evaluation including hyperparameter tuning.'
)
def squad_pipeline():
    # Retrieve bucket name from Kubernetes secrets
    bucket_name = dsl.PipelineParam(
        name='squad-bucket',
        value='coms-6156-kubeflow'  
    )
    
    # Download SQuAD dataset
    download_task = download_squad_data()
    
    # Perform data quality checks
    quality_check_task = squad_data_quality_check(file_path=download_task.output)
    
    # Preprocess the downloaded data
    preprocess_task = process_squad_data(squad_path=download_task.output)
    
    # Train the model
    train_task = train_model(
        data_dir=preprocess_task.output,
        model_dir=f'gs://{bucket_name}/models/saved_models',
        output_dir=f'gs://{bucket_name}/models/saved_models',
        train_file='squad-kubeflow.json'
    )
    
    # Evaluate the model
    evaluate_task = evaluate_model(
        model_dir=train_task.output,
        data_dir=preprocess_task.output,
        result_path=f'gs://{bucket_name}/models/saved_models/evaluation/results'
    )

    # Launch Katib experiment for hyperparameter tuning
    katib_task = katib_launcher(
        experiment_file='./src/automl/katib_squad_grid.yaml'
    )
    
    # Fetch the best hyperparameters
    fetch_hyperparameters_task = fetch_best_hyperparameters(
        experiment_name='katib_experiment_name',
        output_path='./src/automl/squad_best_hyperparameters.json'
    )

    # Retrain the model with the best hyperparameters
    retrain_task = retrain_model(
        data_dir=preprocess_task.output,
        model_dir=f'gs://{bucket_name}/models/saved_models/retrained_model',
        hyperparameters=fetch_hyperparameters_task.output
    )

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(pipeline_func=squad_pipeline, package_path='squad_pipeline.yaml')
