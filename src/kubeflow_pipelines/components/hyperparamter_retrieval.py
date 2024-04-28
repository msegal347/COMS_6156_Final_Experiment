import kfp
from kfp.v2.dsl import component, OutputPath

@component
def fetch_best_hyperparameters(
    experiment_name: str,  
    output_path: OutputPath(str),  
    namespace: str = "kubeflow"  
):
    """Fetch the best hyperparameters from a completed Katib experiment."""
    from kubernetes import client, config
    import json

    config.load_kube_config()  
    api_instance = client.CustomObjectsApi()

    try:
        # Fetch the experiment
        exp = api_instance.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            name=experiment_name
        )
        # Extract the best trial's parameters
        best_trial = exp['status']['currentOptimalTrial']['parameterAssignments']
        params = {p['name']: p['value'] for p in best_trial}
        
        with open(output_path, "w") as f:
            json.dump(params, f)
        
        print("Best hyperparameters:", params)
    except Exception as e:
        print(f"Failed to fetch experiment details: {str(e)}")
        raise
