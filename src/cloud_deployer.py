import os
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import boto3
from botocore.exceptions import ClientError
from azure.storage.blob import BlobServiceClient
from google.cloud.storage import Client as GCPStorageClient
import re

class CloudDeployer:
    """Handles model deployment to different cloud providers."""

    def __init__(self):
        self.mlflow_client = MlflowClient()
        self.cloud_provider = os.getenv('CLOUD_PROVIDER', 'aws').lower()
        self.default_bucket_name = 'mlops-pipeline-demo'

    def _sanitize_bucket_name(self, bucket_name):
        """Sanitize bucket name to comply with S3 naming rules."""
        # Convert to lowercase and replace invalid characters with hyphens
        sanitized = re.sub(r'[^a-z0-9-]', '-', bucket_name.lower())
        # Remove consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        return sanitized

    def _create_bucket_if_not_exists(self, bucket_name=None):
        """Create an S3 bucket if it doesn't exist."""
        if bucket_name is None:
            bucket_name = self.default_bucket_name

        bucket_name = self._sanitize_bucket_name(bucket_name)
        s3_client = boto3.client('s3')

        try:
            s3_client.head_bucket(Bucket=bucket_name)
            return bucket_name, "Bucket already exists"
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:  # Bucket doesn't exist
                try:
                    s3_client.create_bucket(Bucket=bucket_name)
                    return bucket_name, "Bucket created successfully"
                except ClientError as create_error:
                    if 'BucketAlreadyExists' in str(create_error):
                        # If the bucket name is taken, append a timestamp
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        new_bucket_name = f"{bucket_name}-{timestamp}"
                        s3_client.create_bucket(Bucket=new_bucket_name)
                        return new_bucket_name, f"Created bucket with timestamp: {new_bucket_name}"
                    raise Exception(f"Failed to create bucket: {str(create_error)}")
            raise Exception(f"Error checking bucket: {str(e)}")

    def _get_model_file_path(self, run_id):
        """Get the path to the actual model file from MLflow artifacts."""
        artifact_uri = self.mlflow_client.get_run(run_id).info.artifact_uri
        if artifact_uri.startswith('file://'):
            artifact_uri = artifact_uri[7:]

        # The model should be in the MLflow format under the artifacts directory
        model_path = os.path.join(artifact_uri, 'model.pkl')
        if not os.path.exists(model_path):
            # Try alternative model file locations
            model_paths = [
                os.path.join(artifact_uri, 'random_forest_model', 'model.pkl'),
                os.path.join(artifact_uri, 'model', 'model.pkl')
            ]
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                raise FileNotFoundError(f"Could not find model file in {artifact_uri}")

        return model_path

    def deploy_to_aws(self, model_path, bucket_name=None):
        """Deploy model to AWS S3 with enhanced error handling."""
        if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
            raise ValueError("AWS credentials not found in environment variables")

        try:
            # Create or verify bucket exists
            actual_bucket_name, bucket_message = self._create_bucket_if_not_exists(bucket_name)

            s3_client = boto3.client('s3')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"models/churn_prediction_{timestamp}"

            # Upload the model using a file object to ensure seekability
            if not os.path.isfile(model_path):
                raise ValueError(f"Invalid model path: {model_path}")

            with open(model_path, 'rb') as model_file:
                s3_client.upload_fileobj(model_file, actual_bucket_name, s3_key)

            deployment_url = f"s3://{actual_bucket_name}/{s3_key}"
            return {
                'status': 'success',
                'deployment_url': deployment_url,
                'bucket_message': bucket_message
            }
        except ClientError as e:
            raise Exception(f"AWS S3 error: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to deploy to AWS: {str(e)}")

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
        """Deploy the latest model to the configured cloud provider with enhanced validation."""
        if cloud_settings is None:
            cloud_settings = {}

        # Get latest model run
        latest_run = self._get_latest_model()
        model_path = self._get_model_file_path(latest_run.info.run_id)

        # Deploy based on configured cloud provider
        if self.cloud_provider == 'aws':
            bucket_name = cloud_settings.get('bucket_name')
            result = self.deploy_to_aws(model_path, bucket_name)
            return result
        elif self.cloud_provider == 'azure':
            if 'container_name' not in cloud_settings:
                raise ValueError("Azure container name is required")
            return self.deploy_to_azure(model_path, cloud_settings['container_name'])
        elif self.cloud_provider == 'gcp':
            if 'bucket_name' not in cloud_settings:
                raise ValueError("GCP bucket name is required")
            return self.deploy_to_gcp(model_path, cloud_settings['bucket_name'])
        else:
            raise ValueError(f"Unsupported cloud provider: {self.cloud_provider}")