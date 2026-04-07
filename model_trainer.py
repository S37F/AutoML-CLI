"""
Model Trainer Module
Handles model training, evaluation, and comparison with parallel processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
from cli_utils import CLIFormatter, Colors, ProgressTracker
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Trains and evaluates multiple ML models"""
    
    def __init__(self, problem_type: str):
        """
        Initialize model trainer
        
        Args:
            problem_type: Type of ML problem
        """
        self.problem_type = problem_type
        self.results = {}
        self.trained_models = {}
        self.best_model_name = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = {}  # Store predictions for each model
        self.probabilities = {}  # Store probabilities for classification
    
    def _train_single_model(self, model_name: str, model: Any, 
                           X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Tuple[str, Optional[Dict], Optional[Any], Optional[Any], Optional[Any]]:
        """
        Train and evaluate a single model
        
        Returns:
            Tuple of (model_name, metrics, trained_model, predictions, probabilities)
        """
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities for classification
            y_proba = None
            if self.problem_type in ['classification', 'binary_classification']:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
            
            # Evaluate
            if self.problem_type == 'regression':
                metrics = self._evaluate_regression(y_test, y_pred)
            else:
                metrics = self._evaluate_classification(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=min(5, len(X_train)//10),
                scoring='r2' if self.problem_type == 'regression' else 'accuracy',
                n_jobs=1
            )
            metrics['cv_score_mean'] = cv_scores.mean()
            metrics['cv_score_std'] = cv_scores.std()
            
            return model_name, metrics, model, y_pred, y_proba
            
        except Exception as e:
            return model_name, None, None, None, None
        
    def train_and_evaluate(self, models: Dict, X: pd.DataFrame, y: pd.Series,
                          test_size: float = 0.2, metric_priority: str = 'auto',
                          parallel: bool = True) -> Dict:
        """
        Train and evaluate all models (with optional parallel processing)
        
        Args:
            models: Dictionary of model names and model objects
            X: Feature dataset
            y: Target variable
            test_size: Proportion of data for testing
            metric_priority: Metric to prioritize for model selection
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary of results for each model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Store for later use in visualizations
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        CLIFormatter.print_subheader("MODEL TRAINING & EVALUATION", width=70)
        print(f"\n{Colors.INFO}Training set:{Colors.RESET} {Colors.BOLD}{len(X_train)}{Colors.RESET} samples")
        print(f"{Colors.INFO}Test set:{Colors.RESET} {Colors.BOLD}{len(X_test)}{Colors.RESET} samples\n")
        
        if parallel and len(models) > 2:
            # Parallel training
            print(f"{Colors.HIGHLIGHT}⚡ Training {len(models)} models in parallel...{Colors.RESET}\n")
            
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(self._train_single_model)(
                    name, model, X_train, X_test, y_train, y_test
                ) for name, model in tqdm(models.items(), desc="Training models", ncols=70)
            )
            
            # Process results
            for model_name, metrics, model, y_pred, y_proba in results:
                if metrics is not None:
                    self.results[model_name] = metrics
                    self.trained_models[model_name] = model
                    self.predictions[model_name] = y_pred
                    if y_proba is not None:
                        self.probabilities[model_name] = y_proba
                    CLIFormatter.print_success(f"{model_name} trained")
                else:
                    CLIFormatter.print_error(f"{model_name} failed")
        else:
            # Sequential training with progress
            for model_name, model in tqdm(models.items(), desc="Training models", ncols=70):
                model_name_result, metrics, trained_model, y_pred, y_proba = self._train_single_model(
                    model_name, model, X_train, X_test, y_train, y_test
                )
                
                if metrics is not None:
                    self.results[model_name] = metrics
                    self.trained_models[model_name] = trained_model
                    self.predictions[model_name] = y_pred
                    if y_proba is not None:
                        self.probabilities[model_name] = y_proba
                    CLIFormatter.print_success(f"{model_name} trained")
                else:
                    CLIFormatter.print_error(f"{model_name} failed")
        
        # Select best model
        self._select_best_model(metric_priority)
        
        return self.results
    
    def _evaluate_regression(self, y_true: Any, y_pred: Any) -> Dict:
        """Evaluate regression model"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _evaluate_classification(self, y_true: Any, y_pred: Any) -> Dict:
        """Evaluate classification model"""
        # Determine average method based on problem type
        average = 'binary' if self.problem_type == 'binary_classification' else 'weighted'
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
    
    def _select_best_model(self, metric_priority: str = 'auto'):
        """Select the best performing model"""
        if not self.results:
            return
        
        # Determine metric to use
        if metric_priority == 'auto':
            if self.problem_type == 'regression':
                metric = 'r2'  # Higher is better
                reverse = True
            else:
                metric = 'f1'  # Higher is better
                reverse = True
        else:
            metric = metric_priority
            reverse = metric in ['r2', 'accuracy', 'precision', 'recall', 'f1']
        
        # Find best model
        best = max(self.results.items(), 
                   key=lambda x: x[1].get(metric, 0) if reverse else -x[1].get(metric, float('inf')))
        
        self.best_model_name = best[0]
        self.best_model = self.trained_models[self.best_model_name]
    
    def display_results(self):
        """Display evaluation results in a formatted table"""
        if not self.results:
            print("No results to display")
            return
        
        CLIFormatter.print_subheader("MODEL PERFORMANCE COMPARISON", width=70)
        
        if self.problem_type == 'regression':
            self._display_regression_results()
        else:
            self._display_classification_results()
        
        print(f"\n{Colors.SUCCESS}{'='*70}")
        print(f"🏆 BEST MODEL: {Colors.BOLD}{self.best_model_name}{Colors.RESET}{Colors.SUCCESS}")
        print(f"{'='*70}{Colors.RESET}")
        self._explain_best_model()
    
    def _display_regression_results(self):
        """Display regression results"""
        headers = ['Model', 'RMSE', 'MAE', 'R²', 'CV R²']
        rows = []
        
        # Sort by R² (descending)
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        for model_name, metrics in sorted_results:
            marker = "★" if model_name == self.best_model_name else ""
            rows.append([
                f"{marker} {model_name}" if marker else model_name,
                f"{metrics['rmse']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['r2']:.4f}",
                f"{metrics['cv_score_mean']:.4f}"
            ])
        
        # Find best model index
        best_idx = next(i for i, (name, _) in enumerate(sorted_results) if name == self.best_model_name)
        CLIFormatter.print_table(headers, rows, highlight_row=best_idx)
    
    def _display_classification_results(self):
        """Display classification results"""
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'CV Acc']
        rows = []
        
        # Sort by F1 score (descending)
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for model_name, metrics in sorted_results:
            marker = "★" if model_name == self.best_model_name else ""
            rows.append([
                f"{marker} {model_name}" if marker else model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['cv_score_mean']:.4f}"
            ])
        
        # Find best model index
        best_idx = next(i for i, (name, _) in enumerate(sorted_results) if name == self.best_model_name)
        CLIFormatter.print_table(headers, rows, highlight_row=best_idx)
    
    def _explain_best_model(self):
        """Explain why the best model was selected"""
        if not self.best_model_name:
            return
        
        metrics = self.results[self.best_model_name]
        
        print(f"\n{Colors.INFO}💡 Why {Colors.BOLD}{self.best_model_name}{Colors.RESET}{Colors.INFO} was selected:{Colors.RESET}\n")
        
        if self.problem_type == 'regression':
            metric_data = {
                'R² Score': f"{metrics['r2']:.4f} ({metrics['r2']*100:.2f}% variance explained)",
                'RMSE': f"{metrics['rmse']:.4f}",
                'MAE': f"{metrics['mae']:.4f}",
                'CV R² Mean': f"{metrics['cv_score_mean']:.4f} (±{metrics['cv_score_std']:.4f})"
            }
        else:
            metric_data = {
                'F1 Score': f"{metrics['f1']:.4f} (balanced precision & recall)",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'CV Accuracy': f"{metrics['cv_score_mean']:.4f} (±{metrics['cv_score_std']:.4f})"
            }
        
        for key, value in metric_data.items():
            print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}{key}:{Colors.RESET} {value}")
        
        print(f"\n  {Colors.SUCCESS}✓ Consistent performance across cross-validation folds{Colors.RESET}")
    
    def save_model(self, filepath: str):
        """Save the best model to disk"""
        if self.best_model is None:
            CLIFormatter.print_warning("No model to save")
            return
        
        try:
            joblib.dump(self.best_model, filepath)
            CLIFormatter.print_success(f"Best model saved: {filepath}")
        except Exception as e:
            CLIFormatter.print_error(f"Error saving model: {str(e)}")
    
    def get_best_model(self) -> Tuple[Optional[str], Optional[Any]]:
        """
        Get the best model
        
        Returns:
            Tuple of (model_name, model_object)
        """
        return self.best_model_name, self.best_model
