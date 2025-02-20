import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

class ModelMonitor:
    def __init__(self):
        self.predictions_log = []
        self.metrics_log = []
        
    def log_prediction(self, prediction, actual=None, timestamp=None):
        """Log a single prediction with optional actual value."""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.predictions_log.append({
            'timestamp': timestamp,
            'prediction': prediction,
            'actual': actual
        })
        
    def log_metrics(self, metrics, timestamp=None):
        """Log model metrics."""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.metrics_log.append({
            'timestamp': timestamp,
            **metrics
        })
        
    def get_prediction_distribution(self):
        """Create prediction distribution plot."""
        df = pd.DataFrame(self.predictions_log)
        
        fig = px.histogram(
            df,
            x='prediction',
            title='Prediction Distribution',
            nbins=20
        )
        
        return fig
    
    def get_metrics_trend(self):
        """Create metrics trend plot."""
        df = pd.DataFrame(self.metrics_log)
        
        fig = px.line(
            df,
            x='timestamp',
            y=['accuracy', 'precision', 'recall', 'f1'],
            title='Model Metrics Over Time'
        )
        
        return fig
