import streamlit as st
import pandas as pd
from src.data_generator import generate_synthetic_data
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.model_monitor import ModelMonitor
import os
from dotenv import load_dotenv
import mlflow
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

def test_aws_credentials(access_key, secret_key):
    """Test if AWS credentials are valid."""
    try:
        client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        client.list_buckets()
        return True, None
    except ClientError as e:
        return False, str(e)

def main():
    st.set_page_config(page_title="MLOps Pipeline Demo", layout="wide")

    st.title("MLOps Pipeline Demo: Customer Churn Prediction")

    # Initialize session state
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ModelTrainer()
    if 'model_monitor' not in st.session_state:
        st.session_state.model_monitor = ModelMonitor()
    if 'aws_credentials_valid' not in st.session_state:
        st.session_state.aws_credentials_valid = False

    # Sidebar
    st.sidebar.title("Pipeline Steps")
    step = st.sidebar.radio(
        "Select Step",
        ["Data Generation", "Data Processing", "Model Training", "Model Evaluation", "Model Monitoring", "Cloud Deployment"]
    )

    if step == "Data Generation":
        st.header("Data Generation")
        n_samples = st.slider("Number of samples", 100, 5000, 1000)

        if st.button("Generate Data"):
            data = generate_synthetic_data(n_samples)
            st.session_state.data = data
            st.write("Generated Data Preview:")
            st.dataframe(data.head())

            st.write("Data Statistics:")
            st.dataframe(data.describe())


    elif step == "Data Processing":
        if 'data' not in st.session_state:
            st.error("Please generate data first!")
            return

        st.header("Data Processing")

        if st.button("Process Data"):
            processed_data = st.session_state.data_processor.preprocess_data(st.session_state.data)
            X_train, X_test, y_train, y_test = st.session_state.data_processor.split_data(processed_data)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            st.write("Processed Data Preview:")
            st.dataframe(processed_data.head())

            st.write("Training set shape:", X_train.shape)
            st.write("Testing set shape:", X_test.shape)


    elif step == "Model Training":
        if 'X_train' not in st.session_state:
            st.error("Please process data first!")
            return

        st.header("Model Training")

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = st.session_state.model_trainer.train_model(
                    st.session_state.X_train,
                    st.session_state.y_train
                )

                metrics = st.session_state.model_trainer.evaluate_model(
                    st.session_state.X_test,
                    st.session_state.y_test
                )

                st.session_state.model = model
                st.session_state.metrics = metrics

                st.success("Model trained successfully!")
                st.write("Model Metrics:")
                st.json(metrics)


    elif step == "Model Evaluation":
        if 'model' not in st.session_state:
            st.error("Please train model first!")
            return

        st.header("Model Evaluation")

        evaluator = ModelEvaluator(
            st.session_state.model,
            st.session_state.X_test,
            st.session_state.y_test
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            st.plotly_chart(evaluator.plot_confusion_matrix())

        with col2:
            st.subheader("ROC Curve")
            st.plotly_chart(evaluator.plot_roc_curve())

        st.subheader("Feature Importance")
        st.plotly_chart(evaluator.plot_feature_importance())


    elif step == "Model Monitoring":
        if 'model' not in st.session_state:
            st.error("Please train model first!")
            return

        st.header("Model Monitoring")

        # Log some sample predictions
        if len(st.session_state.model_monitor.predictions_log) == 0:
            predictions = st.session_state.model.predict(st.session_state.X_test[:100])
            for pred in predictions:
                st.session_state.model_monitor.log_prediction(pred)

            st.session_state.model_monitor.log_metrics(st.session_state.metrics)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Distribution")
            st.plotly_chart(st.session_state.model_monitor.get_prediction_distribution())

        with col2:
            st.subheader("Metrics Trend")
            st.plotly_chart(st.session_state.model_monitor.get_metrics_trend())


    elif step == "Cloud Deployment":
        if 'model' not in st.session_state:
            st.error("Please train a model first!")
            return

        st.header("Cloud Deployment")

        # Cloud provider selection
        cloud_provider = st.selectbox(
            "Select Cloud Provider",
            ["AWS", "Azure", "GCP"],
            key="cloud_provider"
        )

        if cloud_provider == "AWS":
            st.subheader("AWS Configuration")

            # AWS Credentials Section
            st.markdown("### AWS Credentials")
            with st.expander("Configure AWS Credentials"):
                st.markdown("""
                Your AWS credentials are used to securely deploy models to AWS S3.
                These credentials will be stored securely in your environment.
                """)

                aws_access_key = st.text_input("AWS Access Key ID", 
                    value=os.getenv('AWS_ACCESS_KEY_ID', ''),
                    type="password")
                aws_secret_key = st.text_input("AWS Secret Access Key",
                    value=os.getenv('AWS_SECRET_ACCESS_KEY', ''),
                    type="password")

                if st.button("Validate AWS Credentials"):
                    if aws_access_key and aws_secret_key:
                        with st.spinner("Validating AWS credentials..."):
                            is_valid, error = test_aws_credentials(aws_access_key, aws_secret_key)
                            if is_valid:
                                st.success("AWS credentials are valid!")
                                os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
                                os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
                                st.session_state.aws_credentials_valid = True
                            else:
                                st.error(f"Invalid AWS credentials: {error}")
                                st.session_state.aws_credentials_valid = False
                    else:
                        st.error("Please enter both AWS Access Key ID and Secret Access Key")

            # S3 Bucket Configuration
            st.markdown("### S3 Bucket Configuration")
            bucket_name = st.text_input(
                "S3 Bucket Name (Optional)", 
                os.getenv('AWS_BUCKET_NAME', ''),
                help="Leave empty to use default project bucket name"
            )

            # Deployment Button
            if st.button("Deploy Model"):
                if not st.session_state.aws_credentials_valid:
                    st.error("Please validate your AWS credentials first")
                    return

                with st.spinner("Deploying model to AWS S3..."):
                    try:
                        os.environ['CLOUD_PROVIDER'] = cloud_provider.lower()
                        cloud_settings = {'bucket_name': bucket_name} if bucket_name else {}
                        result = st.session_state.model_trainer.deploy_model(cloud_settings)

                        if result.get('status') == 'success':
                            st.success("Model deployed successfully!")
                            if result.get('bucket_message'):
                                st.info(result['bucket_message'])
                            st.info(f"Deployment URL: {result['deployment_url']}")

                            # Add deployment metadata to MLflow
                            with mlflow.start_run(run_name="model_deployment"):
                                mlflow.log_param("cloud_provider", cloud_provider)
                                mlflow.log_param("deployment_url", result['deployment_url'])
                        else:
                            st.error(f"Deployment failed: {result['message']}")
                    except Exception as e:
                        st.error(f"Deployment error: {str(e)}")

        elif cloud_provider == "Azure":
            cloud_settings = {'container_name': st.text_input("Storage Container Name", os.getenv('AZURE_CONTAINER_NAME', ''))}
        elif cloud_provider == "GCP":
            cloud_settings = {'bucket_name': st.text_input("GCS Bucket Name", os.getenv('GCP_BUCKET_NAME', ''))}


if __name__ == "__main__":
    main()