# COMS 6156 Final Project README

## Enhancing MLOps with Kubeflow, GitHub Actions, AutoML, and SHAP

This repository contains the code and documentation for the final project of COMS 6156. The project aims to enhance the MLOps pipeline of a machine learning model by integrating Kubeflow, GitHub Actions, AutoML, and SHAP.

Pipeline deployment code for both the ResNet-18 and BERT-SQuAD models are in the deployments directory.

Initial experimentation with SHAP is stored in the experiments directory.

The code to train the basic models is in the models directory, and is directly adapted from:

https://github.com/kamalkraj/BERT-SQuAD/tree/master

https://huggingface.co/microsoft/resnet-18

### Setup

To run the code, you will need to install the necessary dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

### Running the Code

Most of the code is specifically configured to run within Kubeflow. To configure the environment correctly for Google Cloud:

1. Create a Google Cloud project
2. Configure and launch a Google Virtual Machine (VM). 
3. Run through the Kubeflow installation [14] (Oauth, Deploy management cluster, Deploy Kubeflow cluster). This may require multiple attempts, as Kubeflow services handle the provisioning of the Kubeflow resources, and may not be readily accessible.
3. Create a Dockerfile for the Docker containers that will be used for the Kubeflow Pipeline 
4. Create a Google Storage bucket for the project
5. Compress the Dockerfile (tar.gz)
“tar -czvf Dockerfile.tar.gz Dockerfile”
6. Submit the Dockerfile to the Google Storage Bucket

Once Kubeflow is installed, you will also need to configure an additional Kubernetes node pool with the necessary compute resources (at least 1 GPU) to run the pipeline. The default Kubeflow node pool does not have the necessary resources to run the pipeline.

Once Kubeflow is installed and the necessary resources are configured, you initialize a notebook inside of Kubeflow, clone this repository, and run the code contained within the notebooks in the src/kubeflow_pipelines/pipelines directory. This will generate the YAML files for the pipeline, which can then be selected within Kubeflow and run.

### Code Structure

The src/automl directory contains the code for the AutoML Grid Search runs for the ResNet-18 and BERT-SQuAD models.

The src/data_management directory contains the necessary code to download, preprocess, and conduct data quality checks on both the ImageNet and SQuAD datasets.

The src/kubeflow_pipelines/components directory contains the code for the custom Kubeflow components used in the pipeline, and includes all of the component code for both models.

The src/kubeflow_pipelines/pipelines directory contains the code for the Kubeflow pipeline for both models.

The tests/unit_tests directory contains the unit tests for the data management functions.

