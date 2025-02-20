import os
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud.storage import Client as GCPStorageClient

class CloudDeployer:
    """Handles model deployment to different cloud providers."""
    
    def __init__(self):
        self.mlflow_client = MlflowClient()
        self.cloud_provider = os.getenv('CLOUD_PROVIDER', 'aws').lower()
        
    def _get_latest_model(self, experiment_name="churn_prediction"):
        """Get the latest model from MLflow."""
        experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")
            
        runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            raise ValueError("No runs found for the experiment")
            
        return runs[0]
        
    def deploy_to_aws(self, model_path, bucket_name):
        """Deploy model to AWS S3."""
        s3_client = boto3.client('s3')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"models/churn_prediction_{timestamp}"
        
        try:
            s3_client.upload_file(model_path, bucket_name, s3_key)
            return f"s3://{bucket_name}/{s3_key}"
        except Exception as e:
            raise Exception(f"Failed to deploy to AWS: {str(e)}")
            
    def deploy_to_azure(self, model_path, container_name):
        """Deploy model to Azure Blob Storage."""
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("Azure storage connection string not found")
            
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"models/churn_prediction_{timestamp}"
        
        try:
            with open(model_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data)
            return f"azure://{container_name}/{blob_name}"
        except Exception as e:
            raise Exception(f"Failed to deploy to Azure: {str(e)}")
            
    def deploy_to_gcp(self, model_path, bucket_name):
        """Deploy model to Google Cloud Storage."""
        storage_client = GCPStorageClient()
        bucket = storage_client.bucket(bucket_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"models/churn_prediction_{timestamp}"
        
        try:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(model_path)
            return f"gs://{bucket_name}/{blob_name}"
        except Exception as e:
            raise Exception(f"Failed to deploy to GCP: {str(e)}")
            
    def deploy_model(self, cloud_settings=None):
        """Deploy the latest model to the configured cloud provider."""
        if cloud_settings is None:
            cloud_settings = {}
            
        # Get latest model run
        latest_run = self._get_latest_model()
        model_path = mlflow.get_run(latest_run.info.run_id).info.artifact_uri
        
        # Deploy based on configured cloud provider
        if self.cloud_provider == 'aws':
            bucket_name = cloud_settings.get('bucket_name', os.getenv('AWS_BUCKET_NAME'))
            return self.deploy_to_aws(model_path, bucket_name)
        elif self.cloud_provider == 'azure':
            container_name = cloud_settings.get('container_name', os.getenv('AZURE_CONTAINER_NAME'))
            return self.deploy_to_azure(model_path, container_name)
        elif self.cloud_provider == 'gcp':
            bucket_name = cloud_settings.get('bucket_name', os.getenv('GCP_BUCKET_NAME'))
            return self.deploy_to_gcp(model_path, bucket_name)
        else:
            raise ValueError(f"Unsupported cloud provider: {self.cloud_provider}")
