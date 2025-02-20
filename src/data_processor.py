import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.numeric_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = ['ContractType', 'InternetService', 'PaymentMethod']
        self.target = 'Churn'
        self.scalers = {}
        self.encoders = {}
        
    def preprocess_data(self, data, is_training=True):
        """Preprocess the data by scaling numerical features and encoding categorical features."""
        processed_data = data.copy()
        
        # Scale numeric features
        for feature in self.numeric_features:
            if is_training:
                self.scalers[feature] = StandardScaler()
                processed_data[feature] = self.scalers[feature].fit_transform(processed_data[[feature]])
            else:
                processed_data[feature] = self.scalers[feature].transform(processed_data[[feature]])
        
        # Encode categorical features
        for feature in self.categorical_features:
            if is_training:
                self.encoders[feature] = LabelEncoder()
                processed_data[feature] = self.encoders[feature].fit_transform(processed_data[feature])
            else:
                processed_data[feature] = self.encoders[feature].transform(processed_data[feature])
        
        return processed_data
    
    def split_data(self, data, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X = data.drop(columns=[self.target])
        y = data[self.target]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
