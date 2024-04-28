import kfp
from kfp import dsl
from kfp.dsl import InputPath, OutputPath
from kubernetes.client.models import V1EnvFromSource, V1SecretKeySelector

from kubeflow_pipelines.components.imagenet.imagenet_downloader import download_tiny_imagenet
from kubeflow_pipelines.components.imagenet.imagenet_data_preprocessor import process_imagenet_data
from kubeflow_pipelines.components.imagenet.imagenet_data_quality import imagenet_data_quality_check
from kubeflow_pipelines.components.imagenet.imagenet_model_train import train_imagenet_model
from kubeflow_pipelines.components.imagenet.imagenet_model_evaluation import validate_imagenet_model
from kubeflow_pipelines.components.katib_launcher import katib_launcher
from kubeflow_pipelines.components.hyperparamter_retrieval import fetch_best_hyperparameters
from kubeflow_pipelines.components.retrain_model import retrain_model


@dsl.pipeline(
    name='imagenet-training-pipeline',
    description='A pipeline that handles Imagenet dataset from download to evaluation.'
)
def imagenet_pipeline():
    # Retrieve bucket name from Kubernetes secrets
    bucket_name = dsl.PipelineParam(
        name='imagenet-bucket',
        value='coms-6156-kubeflow'  
    )
    
    # Download Imagenet dataset
    download_task = download_tiny_imagenet()
    
    # Perform data quality checks
    quality_check_task = imagenet_data_quality_check(file_path=download_task.output)
    
    # Preprocess the downloaded data
    preprocess_task = process_imagenet_data(squad_path=download_task.output)
    
    # Train the model
    train_task = train_imagenet_model(
        data_dir=preprocess_task.output,
        model_dir=f'gs://{bucket_name}/models/saved_models',
        output_dir=f'gs://{bucket_name}/models/saved_models',
        train_file='squad-kubeflow.json'
    )
    
    # Evaluate the model
    evaluate_task = validate_imagenet_model(
        model_dir=train_task.output,
        data_dir=preprocess_task.output,
        result_path=f'gs://{bucket_name}/models/saved_models/evaluation/results'
    )

    # Launch Katib experiment for hyperparameter tuning
    katib_task = katib_launcher(
        experiment_file='./src/automl/katib_imagenet_grid.yaml'
    )
    
    # Fetch the best hyperparameters
    fetch_hyperparameters_task = fetch_best_hyperparameters(
        experiment_name='katib_experiment_name',
        output_path='./src/automl/imagenet_best_hyperparameters.json'
    )

    # Retrain the model with the best hyperparameters
    retrain_task = retrain_model(
        data_dir=preprocess_task.output,
        model_dir=f'gs://{bucket_name}/models/saved_models/retrained_model',
        hyperparameters=fetch_hyperparameters_task.output
    )


if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(pipeline_func=imagenet_pipeline, package_path='imagenet_pipeline.yaml')
