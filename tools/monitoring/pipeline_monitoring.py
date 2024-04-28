import kfp
from kfp.v2.google.client import AIPlatformClient
import os
import datetime

def get_kfp_client():
    """Initialize the KFP client."""
    kfp_endpoint = os.getenv('KFP_ENDPOINT', 'http://kubeflow.endpoints.coms-6156-kubeflow.cloud.goog/')
    return kfp.Client(host=kfp_endpoint)

def list_recent_pipeline_runs(namespace='kubeflow', count=10):
    """
    List the most recent pipeline runs.
    """
    kfp_client = get_kfp_client()
    runs = kfp_client.runs.list_runs(page_size=count, namespace=namespace).runs
    return runs

def monitor_pipeline_runs():
    """
    Monitor pipeline runs and report their status.
    """
    runs = list_recent_pipeline_runs()
    kfp_client = get_kfp_client()
    if not runs:
        print("No recent runs found.")
        return

    for run in runs:
        run_details = kfp_client.runs.get_run(run_id=run.id)
        status = run_details.run.status
        print(f"Run ID: {run.id}, Name: {run.name}, Status: {status}")
        
        # Check if the run has failed or succeeded
        if status == "Failed":
            print(f"Alert: Pipeline run {run.name} (ID: {run.id}) has failed.")
        elif status == "Succeeded":
            print(f"Success: Pipeline run {run.name} (ID: {run.id}) completed successfully.")
        else:
            print(f"Running: Pipeline run {run.name} (ID: {run.id}) is still in progress.")

if __name__ == "__main__":
    print("Monitoring Kubeflow Pipelines...")
    monitor_pipeline_runs()
