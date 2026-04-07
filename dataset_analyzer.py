"""
Dataset Analyzer Module
Handles dataset ingestion, analysis, and problem type identification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from cli_utils import CLIFormatter, Colors


class DatasetAnalyzer:
    """Analyzes datasets to extract characteristics and identify ML problem types"""
    
    def __init__(self, filepath: str):
        """
        Initialize analyzer with dataset path
        
        Args:
            filepath: Path to CSV dataset
        """
        self.filepath = filepath
        self.df = None
        self.analysis = {}
        
    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.filepath)
            CLIFormatter.print_success(
                f"Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns"
            )
            return self.df
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def analyze_dataset(self) -> Dict:
        """
        Perform comprehensive dataset analysis
        
        Returns:
            Dictionary containing dataset characteristics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        analysis = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns),
            'statistics': {}
        }
        
        # Get statistics for numeric columns
        for col in analysis['numeric_columns']:
            analysis['statistics'][col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'unique_count': self.df[col].nunique()
            }
        
        # Get value counts for categorical columns
        for col in analysis['categorical_columns']:
            analysis['statistics'][col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head(5).to_dict()
            }
        
        self.analysis = analysis
        return analysis
    
    def display_analysis(self):
        """Display dataset analysis in a readable format"""
        if not self.analysis:
            self.analyze_dataset()
        
        CLIFormatter.print_subheader("DATASET ANALYSIS", width=70)
        
        # Display shape info
        print(f"\n{Colors.BOLD}Dataset Shape:{Colors.RESET} {Colors.HIGHLIGHT}{self.analysis['shape'][0]} rows × {self.analysis['shape'][1]} columns{Colors.RESET}")
        
        # Numeric columns
        print(f"\n{Colors.INFO}📊 Numeric Columns ({len(self.analysis['numeric_columns'])}){Colors.RESET}")
        for col in self.analysis['numeric_columns']:
            missing = self.analysis['missing_values'][col]
            unique = self.analysis['statistics'][col]['unique_count']
            missing_str = f"{Colors.WARNING}{missing}{Colors.RESET}" if missing > 0 else f"{Colors.SUCCESS}{missing}{Colors.RESET}"
            print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}{col}{Colors.RESET} {Colors.DIM}(unique: {unique}, missing: {missing_str}{Colors.DIM}){Colors.RESET}")
        
        # Categorical columns
        if self.analysis['categorical_columns']:
            print(f"\n{Colors.INFO}📝 Categorical Columns ({len(self.analysis['categorical_columns'])}){Colors.RESET}")
            for col in self.analysis['categorical_columns']:
                missing = self.analysis['missing_values'][col]
                unique = self.analysis['statistics'][col]['unique_count']
                missing_str = f"{Colors.WARNING}{missing}{Colors.RESET}" if missing > 0 else f"{Colors.SUCCESS}{missing}{Colors.RESET}"
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}{col}{Colors.RESET} {Colors.DIM}(unique: {unique}, missing: {missing_str}{Colors.DIM}){Colors.RESET}")
        
        # Missing values summary
        total_missing = sum(self.analysis['missing_values'].values())
        if total_missing > 0:
            CLIFormatter.print_warning(f"Total missing values: {total_missing}")
        else:
            CLIFormatter.print_success("No missing values detected!")
    
    def suggest_target_columns(self) -> List[str]:
        """
        Suggest potential target columns based on dataset characteristics
        
        Returns:
            List of suggested target column names
        """
        if self.df is None:
            return []
        
        suggestions = []
        
        # Common target column names
        common_targets = ['target', 'label', 'class', 'output', 'y', 'price', 'outcome', 'result']
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Check if column name matches common patterns
            if any(target in col_lower for target in common_targets):
                suggestions.append(col)
                continue
            
            # Check if last column (often target in datasets)
            if col == self.df.columns[-1]:
                suggestions.append(col)
    
        return suggestions if suggestions else list(self.df.columns[-3:])
    
    def identify_problem_type(self, target_column: str) -> Tuple[str, str]:
        """
        Identify the ML problem type based on target column
        
        Args:
            target_column: Name of the target column
            
        Returns:
            Tuple of (problem_type, explanation)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if target_column not in self.df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        
        target_data = self.df[target_column]
        unique_count = target_data.nunique()
        data_type = target_data.dtype
        
        # Rule-based problem type identification
        if data_type in ['int64', 'float64']:
            # Check if it's actually categorical (low unique values)
            if unique_count == 2:
                explanation = (
                    f"{Colors.SUCCESS}✓ Binary Classification detected{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Target has exactly 2 unique values: {Colors.HIGHLIGHT}{list(target_data.unique())}{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} This is a binary classification problem"
                )
                return 'binary_classification', explanation
            
            elif unique_count <= 10:
                explanation = (
                    f"{Colors.SUCCESS}✓ Multi-class Classification detected{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Target has {Colors.HIGHLIGHT}{unique_count}{Colors.RESET} unique values: {Colors.HIGHLIGHT}{list(target_data.unique())}{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Small number of discrete values suggests classification"
                )
                return 'classification', explanation
            
            else:
                explanation = (
                    f"{Colors.SUCCESS}✓ Regression detected{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Target is numeric with {Colors.HIGHLIGHT}{unique_count}{Colors.RESET} unique values\n"
                    f"  {Colors.DIM}•{Colors.RESET} Range: {Colors.HIGHLIGHT}[{target_data.min():.2f}, {target_data.max():.2f}]{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Continuous values suggest regression problem"
                )
                return 'regression', explanation
        
        else:  # Categorical/Object type
            if unique_count == 2:
                explanation = (
                    f"{Colors.SUCCESS}✓ Binary Classification detected{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Target has 2 categories: {Colors.HIGHLIGHT}{list(target_data.unique())}{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} This is a binary classification problem"
                )
                return 'binary_classification', explanation
            
            else:
                explanation = (
                    f"{Colors.SUCCESS}✓ Multi-class Classification detected{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Target has {Colors.HIGHLIGHT}{unique_count}{Colors.RESET} categories\n"
                    f"  {Colors.DIM}•{Colors.RESET} Top categories: {Colors.HIGHLIGHT}{list(target_data.value_counts().head(3).index)}{Colors.RESET}\n"
                    f"  {Colors.DIM}•{Colors.RESET} Categorical target suggests classification"
                )
                return 'classification', explanation
    
    def get_feature_target_split(self, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataset into features and target
        
        Args:
            target_column: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if target_column not in self.df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        return X, y
    
    def get_missing_value_info(self) -> Dict:
        """Get detailed information about missing values"""
        if self.df is None:
            return {}
        
        missing_info = {}
        total_rows = len(self.df)
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': (missing_count / total_rows) * 100
                }
        
        return missing_info
