import kfp
from kfp.v2.dsl import component
import kubernetes.client as k8s_client
from kfp.v2 import dsl

@component
def katib_launcher(experiment_file: str):
    """Launch a hyperparameter tuning job using Katib."""
    from kubernetes import client, config
    from kubernetes.client import rest
    import yaml

    config.load_kube_config()  
    api_instance = client.CustomObjectsApi()

    with open(experiment_file, 'r') as f:
        exp = yaml.safe_load(f)

    try:
        api_response = api_instance.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace="kubeflow",
            plural="experiments",
            body=exp,
        )
        print(api_response)
    except rest.ApiException as e:
        print(f"Exception when calling CustomObjectsApi->create_namespaced_custom_object: {e}")
