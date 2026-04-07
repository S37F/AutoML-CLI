"""
Data Preprocessor Module
Handles data preprocessing, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional
from cli_utils import CLIFormatter, Colors


class DataPreprocessor:
    """Preprocesses data for machine learning"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_columns = []
        self.categorical_columns = []
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit and transform the data
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        X_processed = X.copy()
        
        # Identify column types
        self.numeric_columns = list(X_processed.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(X_processed.select_dtypes(exclude=[np.number]).columns)
        
        # Handle missing values
        if self.numeric_columns:
            X_processed[self.numeric_columns] = self.numeric_imputer.fit_transform(
                X_processed[self.numeric_columns]
            )
        
        if self.categorical_columns:
            X_processed[self.categorical_columns] = self.categorical_imputer.fit_transform(
                X_processed[self.categorical_columns]
            )
        
        # Encode categorical variables
        for col in self.categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numeric features
        if self.numeric_columns:
            X_processed[self.numeric_columns] = self.scaler.fit_transform(
                X_processed[self.numeric_columns]
            )
        
        # Encode target if categorical
        y_processed = y
        if y is not None:
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y_processed = pd.Series(le.fit_transform(y), index=y.index, dtype='int64')  # type: ignore
                self.label_encoders['target'] = le
        
        return X_processed, y_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted parameters
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed dataframe
        """
        X_processed = X.copy()
        
        # Handle missing values
        if self.numeric_columns:
            X_processed[self.numeric_columns] = self.numeric_imputer.transform(
                X_processed[self.numeric_columns]
            )
        
        if self.categorical_columns:
            X_processed[self.categorical_columns] = self.categorical_imputer.transform(
                X_processed[self.categorical_columns]
            )
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in self.label_encoders:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Scale numeric features
        if self.numeric_columns:
            X_processed[self.numeric_columns] = self.scaler.transform(
                X_processed[self.numeric_columns]
            )
        
        return X_processed
    
    def get_preprocessing_summary(self) -> str:
        """Get summary of preprocessing steps applied"""
        CLIFormatter.print_subheader("DATA PREPROCESSING SUMMARY", width=70)
        
        print(f"\n{Colors.INFO}🔢 Numeric columns ({len(self.numeric_columns)}):{Colors.RESET}")
        print(f"  {Colors.DIM}•{Colors.RESET} Imputed missing values using {Colors.BOLD}median{Colors.RESET}")
        print(f"  {Colors.DIM}•{Colors.RESET} Standardized features (mean=0, std=1)")
        
        if self.categorical_columns:
            print(f"\n{Colors.INFO}🔤 Categorical columns ({len(self.categorical_columns)}):{Colors.RESET}")
            print(f"  {Colors.DIM}•{Colors.RESET} Imputed missing values using {Colors.BOLD}most frequent{Colors.RESET}")
            print(f"  {Colors.DIM}•{Colors.RESET} Label encoded to numeric values")
        
        return ""
