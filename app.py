import streamlit as st
import pandas as pd
from src.data_generator import generate_synthetic_data
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.model_monitor import ModelMonitor

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
    
    # Sidebar
    st.sidebar.title("Pipeline Steps")
    step = st.sidebar.radio(
        "Select Step",
        ["Data Generation", "Data Processing", "Model Training", "Model Evaluation", "Model Monitoring"]
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

if __name__ == "__main__":
    main()
