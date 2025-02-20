# MLOps Pipeline Demo: Customer Churn Prediction

This project demonstrates an end-to-end MLOps pipeline for customer churn prediction. It showcases various aspects of MLOps, including data generation, processing, model training, evaluation, and monitoring.

## Pipeline Components

### 1. Data Generation
- Generates synthetic customer data with realistic features
- Includes demographic information, service usage, and payment details
- Implements controlled randomization for realistic data distribution

### 2. Data Processing
- Handles numerical and categorical features
- Implements feature scaling and encoding
- Provides train-test split functionality

### 3. Model Training
- Uses Random Forest Classifier for prediction
- Implements MLflow for experiment tracking
- Logs model parameters and metrics

### 4. Model Evaluation
- Provides comprehensive model performance metrics
- Generates interactive visualizations:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance

### 5. Model Monitoring
- Tracks prediction distribution
- Monitors model metrics over time
- Provides real-time monitoring dashboard

## Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Experiment Tracking**: MLflow

## Best Practices

1. **Code Organization**
   - Modular structure with separate components
   - Clear separation of concerns
   - Comprehensive documentation

2. **Data Handling**
   - Proper data validation
   - Scalable preprocessing pipeline
   - Efficient data splitting

3. **Model Management**
   - Experiment tracking
   - Model versioning
   - Performance monitoring

4. **Visualization**
   - Interactive dashboards
   - Real-time monitoring
   - Comprehensive model evaluation

## Getting Started

1. Ensure all required packages are installed
2. Run the application:
   