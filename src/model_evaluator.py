import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_prob = model.predict_proba(X_test)[:, 1]
        
    def plot_confusion_matrix(self):
        """Create confusion matrix plot using plotly."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='RdBu'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        return fig
    
    def plot_roc_curve(self):
        """Create ROC curve plot using plotly."""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure(data=go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        
        return fig
    
    def plot_feature_importance(self):
        """Create feature importance plot using plotly."""
        importance = self.model.feature_importances_
        features = self.X_test.columns
        
        fig = px.bar(
            x=features,
            y=importance,
            title='Feature Importance'
        )
        
        fig.update_layout(
            xaxis_title='Features',
            yaxis_title='Importance Score'
        )
        
        return fig
