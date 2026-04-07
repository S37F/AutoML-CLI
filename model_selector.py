"""
Model Selector Module
Intelligent model selection based on problem type and dataset characteristics
"""

from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
from cli_utils import CLIFormatter, Colors


class ModelSelector:
    """Selects appropriate ML models based on problem type and requirements"""
    
    def __init__(self, problem_type: str):
        """
        Initialize model selector
        
        Args:
            problem_type: Type of ML problem ('regression', 'classification', 'binary_classification')
        """
        self.problem_type = problem_type
        self.selected_models = {}
        
    def select_models(self, interpretability_priority: bool = False, 
                     dataset_size: str = 'medium') -> Dict:
        """
        Select appropriate models based on problem type and preferences
        
        Args:
            interpretability_priority: Whether to prioritize interpretable models
            dataset_size: Size category of dataset ('small', 'medium', 'large')
            
        Returns:
            Dictionary of model names and model objects
        """
        if self.problem_type == 'regression':
            models = self._select_regression_models(interpretability_priority, dataset_size)
        elif self.problem_type in ['classification', 'binary_classification']:
            models = self._select_classification_models(interpretability_priority, dataset_size)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")
        
        self.selected_models = models
        return models
    
    def _select_regression_models(self, interpretability: bool, size: str) -> Dict:
        """Select regression models"""
        models = {}
        
        # Always include Linear Regression (baseline)
        models['Linear Regression'] = LinearRegression()
        
        # Lasso for feature selection
        models['Lasso Regression'] = Lasso(alpha=1.0, random_state=42)
        
        # Ridge regression for regularization
        models['Ridge Regression'] = Ridge(alpha=1.0)
        
        # Polynomial Regression for non-linear relationships
        if size != 'large':  # Skip for large datasets (memory intensive)
            models['Polynomial Regression (degree=2)'] = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('regressor', LinearRegression())
            ])
        
        # Decision Tree - good for interpretability
        models['Decision Tree'] = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        # Random Forest - powerful ensemble method
        n_estimators = 50 if size == 'large' else 100
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - often best performance
        models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        if not interpretability:
            # KNN for non-parametric modeling
            n_neighbors = 5 if size == 'small' else 10
            models['K-Nearest Neighbors'] = KNeighborsRegressor(n_neighbors=n_neighbors)
            
            # SVR for complex patterns (not for large datasets)
            if size != 'large':
                models['Support Vector Regression'] = SVR(kernel='rbf', C=1.0)
            
            # XGBoost - state of the art (if available)
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            
            # LightGBM - fast gradient boosting (if available)
            if LIGHTGBM_AVAILABLE and size != 'small':
                models['LightGBM'] = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
        
        return models
    
    def _select_classification_models(self, interpretability: bool, size: str) -> Dict:
        """Select classification models"""
        models = {}
        
        # Logistic Regression (baseline, interpretable)
        models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        # Naive Bayes - fast and simple
        models['Naive Bayes'] = GaussianNB()
        
        # Decision Tree - interpretable
        models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        # Random Forest - powerful ensemble
        n_estimators = 50 if size == 'large' else 100
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - high performance
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        if not interpretability:
            # KNN
            n_neighbors = 5 if size == 'small' else 10
            models['K-Nearest Neighbors'] = KNeighborsClassifier(n_neighbors=n_neighbors)
            
            # SVM (not for large datasets)
            if size != 'large':
                models['Support Vector Machine'] = SVC(
                    kernel='rbf',
                    C=1.0,
                    probability=True,  # Enable probability for ROC curves
                    random_state=42
                )
            
            # XGBoost (if available)
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            
            # LightGBM (if available)
            if LIGHTGBM_AVAILABLE and size != 'small':
                models['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
        
        return models
    
    def explain_model_selection(self) -> str:
        """
        Provide explanation for why models were selected
        
        Returns:
            Explanation string
        """
        CLIFormatter.print_subheader("MODEL SELECTION REASONING", width=70)
        
        if self.problem_type == 'regression':
            print(f"\n{Colors.INFO}🎯 Regression Models Selected:{Colors.RESET}")
            print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Linear Regression{Colors.RESET} - Baseline model, fast and interpretable")
            
            if 'Lasso Regression' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Lasso Regression{Colors.RESET} - L1 regularization with feature selection")
            
            if 'Ridge Regression' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Ridge Regression{Colors.RESET} - L2 regularization, handles multicollinearity")
            
            if 'Polynomial Regression (degree=2)' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Polynomial Regression{Colors.RESET} - Captures non-linear relationships")
            
            if 'Decision Tree' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Decision Tree{Colors.RESET} - Non-linear patterns, highly interpretable")
            
            if 'Random Forest' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Random Forest{Colors.RESET} - Ensemble method, robust to overfitting")
            
            if 'Gradient Boosting' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Gradient Boosting{Colors.RESET} - Sequential ensemble, high accuracy")
            
            if 'XGBoost' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}XGBoost{Colors.RESET} - State-of-the-art gradient boosting")
            
            if 'LightGBM' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}LightGBM{Colors.RESET} - Fast gradient boosting for large datasets")
            
            if 'K-Nearest Neighbors' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}K-Nearest Neighbors{Colors.RESET} - Non-parametric, local patterns")
            
            if 'Support Vector Regression' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Support Vector Regression{Colors.RESET} - Complex patterns with kernel trick")
        
        else:  # Classification
            print(f"\n{Colors.INFO}🎯 Classification Models Selected:{Colors.RESET}")
            print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Logistic Regression{Colors.RESET} - Baseline classifier, probabilistic outputs")
            
            if 'Naive Bayes' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Naive Bayes{Colors.RESET} - Fast probabilistic classifier")
            
            if 'Decision Tree' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Decision Tree{Colors.RESET} - Rule-based decisions, easy to interpret")
            
            if 'Random Forest' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Random Forest{Colors.RESET} - Ensemble of trees, reduces overfitting")
            
            if 'Gradient Boosting' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Gradient Boosting{Colors.RESET} - Sequential boosting, high performance")
            
            if 'XGBoost' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}XGBoost{Colors.RESET} - Optimized gradient boosting, competition winner")
            
            if 'LightGBM' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}LightGBM{Colors.RESET} - Efficient gradient boosting for large data")
            
            if 'K-Nearest Neighbors' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}K-Nearest Neighbors{Colors.RESET} - Instance-based learning")
            
            if 'Support Vector Machine' in self.selected_models:
                print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BOLD}Support Vector Machine{Colors.RESET} - Complex decision boundaries")
        
        return ""
    
    def get_model_list(self) -> List[str]:
        """Get list of selected model names"""
        return list(self.selected_models.keys())
