import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self):
        self.model = None
        # Set MLflow tracking URI to the local server
        mlflow.set_tracking_uri('http://0.0.0.0:5001')

    def train_model(self, X_train, y_train):
        """Train a Random Forest model and log metrics with MLflow."""
        # Configure MLflow experiment
        experiment_name = "churn_prediction"
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Log model parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)

            # Log the model
            mlflow.sklearn.log_model(self.model, "random_forest_model")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and return metrics."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

        # Log metrics to MLflow if in an active run
        if mlflow.active_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        return metrics