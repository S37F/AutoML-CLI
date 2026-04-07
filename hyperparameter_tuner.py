"""
Hyperparameter Tuning Module
Automated hyperparameter optimization using GridSearchCV and RandomizedSearchCV
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from cli_utils import CLIFormatter, Colors
import warnings
import platform
warnings.filterwarnings('ignore')

# Determine if we're on Windows (has multiprocessing issues)
IS_WINDOWS = platform.system() == 'Windows'

try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb  # type: ignore
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class HyperparameterTuner:
    """Automatically tune hyperparameters for the best model"""
    
    def __init__(self, problem_type: str, cv_folds: int = 5):
        """
        Initialize hyperparameter tuner
        
        Args:
            problem_type: Type of ML problem
            cv_folds: Number of cross-validation folds
        """
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.param_grids = self._get_param_grids()
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for different models"""
        
        # Common parameters for regression and classification
        param_grids = {
            # Random Forest
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            # Gradient Boosting
            'Gradient Boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            
            # Decision Tree
            'Decision Tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8]
            },
            
            # KNN
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        # Regression specific
        if self.problem_type == 'regression':
            param_grids['Ridge Regression'] = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
            param_grids['Lasso Regression'] = {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
            param_grids['Support Vector Regression'] = {
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['rbf', 'linear']
            }
        
        # Classification specific
        else:
            param_grids['Logistic Regression'] = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
            param_grids['Support Vector Machine'] = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 70]
            }
        
        return param_grids
    
    def tune_model(self, model_name: str, model: Any, X_train, y_train, 
                   use_randomized: bool = True, n_iter: int = 20) -> tuple:
        """
        Tune hyperparameters for a specific model
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target
            use_randomized: Use RandomizedSearchCV instead of GridSearchCV
            n_iter: Number of iterations for RandomizedSearchCV
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        if model_name not in self.param_grids:
            CLIFormatter.print_warning(f"No parameter grid defined for {model_name}, skipping tuning")
            return model, {}, None
        
        param_grid = self.param_grids[model_name]
        
        # Determine scoring metric
        if self.problem_type == 'regression':
            scoring = 'r2'
        else:
            scoring = 'f1_weighted'
        
        try:
            print(f"\n{Colors.INFO}⚙ Tuning {model_name}...{Colors.RESET}")
            
            # Use single core on Windows to avoid multiprocessing issues
            n_jobs = 1 if IS_WINDOWS else -1
            
            if use_randomized:
                # RandomizedSearchCV - faster for large parameter spaces
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=self.cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    random_state=42,
                    verbose=0
                )
            else:
                # GridSearchCV - exhaustive search
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=self.cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=0
                )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            CLIFormatter.print_success(
                f"Best {scoring}: {search.best_score_:.4f}"
            )
            
            # Display best parameters
            print(f"{Colors.DIM}  Best parameters:{Colors.RESET}")
            for param, value in search.best_params_.items():
                print(f"{Colors.DIM}    • {param}: {Colors.RESET}{Colors.HIGHLIGHT}{value}{Colors.RESET}")
            
            return search.best_estimator_, search.best_params_, search.best_score_
            
        except Exception as e:
            CLIFormatter.print_error(f"Tuning failed for {model_name}: {str(e)}")
            return model, {}, None
    
    def get_tuning_summary(self, model_name: str, original_score: float, 
                          tuned_score: float, best_params: Dict) -> str:
        """
        Generate a summary of tuning results
        
        Args:
            model_name: Name of the model
            original_score: Score before tuning
            tuned_score: Score after tuning
            best_params: Best parameters found
            
        Returns:
            Summary string
        """
        improvement = ((tuned_score - original_score) / abs(original_score)) * 100 if original_score != 0 else 0
        
        summary = f"\n{Colors.SUCCESS}{'='*70}\n"
        summary += f"HYPERPARAMETER TUNING RESULTS: {model_name}\n"
        summary += f"{'='*70}{Colors.RESET}\n\n"
        
        summary += f"{Colors.INFO}Original Score:{Colors.RESET} {original_score:.4f}\n"
        summary += f"{Colors.SUCCESS}Tuned Score:{Colors.RESET} {Colors.BOLD}{tuned_score:.4f}{Colors.RESET}\n"
        
        if improvement > 0:
            summary += f"{Colors.SUCCESS}Improvement:{Colors.RESET} {Colors.BOLD}+{improvement:.2f}%{Colors.RESET} ✓\n"
        elif improvement < 0:
            summary += f"{Colors.WARNING}Change:{Colors.RESET} {improvement:.2f}%\n"
        else:
            summary += f"{Colors.DIM}No significant change{Colors.RESET}\n"
        
        summary += f"\n{Colors.INFO}Best Parameters:{Colors.RESET}\n"
        for param, value in best_params.items():
            summary += f"  {Colors.DIM}•{Colors.RESET} {param}: {Colors.HIGHLIGHT}{value}{Colors.RESET}\n"
        
        return summary
