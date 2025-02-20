import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic customer churn data."""
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(45, 15, n_samples)
    tenure = np.random.poisson(20, n_samples)
    monthly_charges = np.random.uniform(30, 120, n_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
    contract_type = np.random.choice(['Month-to-month', '1 year', '2 year'], n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], n_samples)
    
    # Generate target (churn)
    churn_prob = 0.3 * (age < 30) + 0.4 * (tenure < 12) + 0.3 * (monthly_charges > 80)
    churn = np.random.binomial(1, churn_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'ContractType': contract_type,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'Churn': churn
    })
    
    return data

if __name__ == "__main__":
    data = generate_synthetic_data()
    print("Generated synthetic data shape:", data.shape)
