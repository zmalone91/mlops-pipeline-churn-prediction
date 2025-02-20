import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def train_model(self, X_train, y_train):
        """Train a Random Forest model and log metrics with MLflow."""
        mlflow.set_tracking_uri('file:./mlruns')
        
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
        
        return metrics
