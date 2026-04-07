"""
Sample dataset generator for testing the AutoML CLI
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate regression dataset (House Prices)
def generate_regression_dataset():
    n_samples = 500
    
    data = {
        'Size_SqFt': np.random.randint(800, 4000, n_samples),
        'Bedrooms': np.random.randint(1, 6, n_samples),
        'Bathrooms': np.random.randint(1, 4, n_samples),
        'Age_Years': np.random.randint(0, 50, n_samples),
        'Distance_City_Miles': np.random.randint(1, 50, n_samples),
        'HasGarage': np.random.choice(['Yes', 'No'], n_samples),
        'Neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate price based on features (with some noise)
    df['Price'] = (
        df['Size_SqFt'] * 150 +
        df['Bedrooms'] * 10000 +
        df['Bathrooms'] * 8000 -
        df['Age_Years'] * 500 -
        df['Distance_City_Miles'] * 200 +
        (df['HasGarage'] == 'Yes') * 15000 +
        np.random.normal(0, 20000, n_samples)
    )
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 20, replace=False), 'Bathrooms'] = np.nan
    df.loc[np.random.choice(df.index, 15, replace=False), 'Distance_City_Miles'] = np.nan
    
    return df

# Generate binary classification dataset (Loan Approval)
def generate_binary_classification_dataset():
    n_samples = 600
    
    data = {
        'Age': np.random.randint(20, 70, n_samples),
        'Income': np.random.randint(20000, 150000, n_samples),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Loan_Amount': np.random.randint(5000, 500000, n_samples),
        'Employment_Years': np.random.randint(0, 40, n_samples),
        'Has_Mortgage': np.random.choice(['Yes', 'No'], n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate approval based on features
    approval_score = (
        (df['Credit_Score'] - 300) / 550 * 0.4 +
        (df['Income'] / 150000) * 0.3 +
        (df['Employment_Years'] / 40) * 0.2 -
        (df['Loan_Amount'] / 500000) * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['Loan_Approved'] = (approval_score > 0.5).astype(int)
    df['Loan_Approved'] = df['Loan_Approved'].map({1: 'Approved', 0: 'Rejected'})
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 25, replace=False), 'Employment_Years'] = np.nan
    df.loc[np.random.choice(df.index, 20, replace=False), 'Credit_Score'] = np.nan
    
    return df

# Generate multi-class classification dataset (Iris-like)
def generate_multiclass_classification_dataset():
    n_samples = 450
    
    data = {
        'Sepal_Length': np.random.uniform(4.0, 8.0, n_samples),
        'Sepal_Width': np.random.uniform(2.0, 4.5, n_samples),
        'Petal_Length': np.random.uniform(1.0, 7.0, n_samples),
        'Petal_Width': np.random.uniform(0.1, 2.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create species based on features
    species = []
    for _, row in df.iterrows():
        if row['Petal_Length'] < 2.5:
            species.append('Setosa')
        elif row['Petal_Length'] < 5.0:
            species.append('Versicolor')
        else:
            species.append('Virginica')
    
    df['Species'] = species
    
    # Add some noise
    noise_indices = np.random.choice(df.index, 50, replace=False)
    df.loc[noise_indices, 'Species'] = np.random.choice(['Setosa', 'Versicolor', 'Virginica'], 50)
    
    return df

# Create sample datasets
if __name__ == "__main__":
    import os
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate and save datasets
    print("Generating sample datasets...")
    
    # Regression dataset
    df_regression = generate_regression_dataset()
    df_regression.to_csv('data/house_prices.csv', index=False)
    print(f"✓ Created: data/house_prices.csv ({len(df_regression)} rows)")
    
    # Binary classification dataset
    df_binary = generate_binary_classification_dataset()
    df_binary.to_csv('data/loan_approval.csv', index=False)
    print(f"✓ Created: data/loan_approval.csv ({len(df_binary)} rows)")
    
    # Multi-class classification dataset
    df_multi = generate_multiclass_classification_dataset()
    df_multi.to_csv('data/iris_classification.csv', index=False)
    print(f"✓ Created: data/iris_classification.csv ({len(df_multi)} rows)")
    
    print("\nSample datasets created successfully!")
    print("\nYou can now test the AutoML CLI with:")
    print("  python automl_cli.py data/house_prices.csv")
    print("  python automl_cli.py data/loan_approval.csv")
    print("  python automl_cli.py data/iris_classification.csv")
