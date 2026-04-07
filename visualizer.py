"""
Visualization Module
Generate comprehensive visualizations for model evaluation
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import learning_curve
import os
from cli_utils import CLIFormatter, Colors

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelVisualizer:
    """Generate visualizations for model evaluation"""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true: Any, y_pred: Any, 
                              class_names: Optional[List] = None,
                              model_name: str = 'Model') -> str:
        """
        Plot confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            
            if class_names is None:
                class_names = [f'Class {i}' for i in range(len(cm))]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'})
            
            plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            return ""
    
    def plot_roc_curve(self, y_true: Any, y_proba: Any, 
                       model_name: str = 'Model',
                       pos_label: Any = 1) -> str:
        """
        Plot ROC curve for binary classification
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model
            pos_label: Label of positive class
            
        Returns:
            Path to saved plot
        """
        try:
            # Handle multi-class by using probability of positive class
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, -1]
            
            fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, f'roc_curve_{model_name.replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating ROC curve: {e}")
            return ""
    
    def plot_residuals(self, y_true: Any, y_pred: Any, 
                       model_name: str = 'Model') -> str:
        """
        Plot residual plots for regression
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        try:
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Residual plot
            axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
            axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Predicted Values', fontsize=12)
            axes[0].set_ylabel('Residuals', fontsize=12)
            axes[0].set_title(f'Residual Plot - {model_name}', fontsize=13, fontweight='bold')
            axes[0].grid(alpha=0.3)
            
            # Distribution of residuals
            axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Residuals', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, f'residuals_{model_name.replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating residual plot: {e}")
            return ""
    
    def plot_learning_curve(self, estimator: Any, X: Any, y: Any,
                           model_name: str = 'Model',
                           cv: int = 5) -> str:
        """
        Plot learning curve to diagnose bias/variance
        
        Args:
            estimator: ML model
            X: Features
            y: Target
            model_name: Name of the model
            cv: Number of CV folds
            
        Returns:
            Path to saved plot
        """
        try:
            result = learning_curve(
                estimator, X, y, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='r2' if hasattr(estimator, 'predict') else 'accuracy',
                random_state=42,
                return_times=False
            )
            train_sizes, train_scores, val_scores = result  # type: ignore
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            
            plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.1, color='r')
            
            plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.1, color='g')
            
            plt.xlabel('Training Set Size', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, f'learning_curve_{model_name.replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating learning curve: {e}")
            return ""
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str = 'Model',
                               top_n: int = 20) -> str:
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_
            feature_names: Names of features
            model_name: Name of the model
            top_n: Number of top features to show
            
        Returns:
            Path to saved plot
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                return ""
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, max(6, top_n * 0.3)))
            
            colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(indices)))  # type: ignore
            
            plt.barh(range(len(indices)), importances[indices], color=colors)
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - {model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, f'feature_importance_{model_name.replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
            return ""
    
    def plot_model_comparison(self, results: Dict[str, Dict],
                            problem_type: str) -> str:
        """
        Plot comparison of all models
        
        Args:
            results: Dictionary of model results
            problem_type: Type of problem
            
        Returns:
            Path to saved plot
        """
        try:
            model_names = list(results.keys())
            
            if problem_type == 'regression':
                metrics = ['r2', 'rmse', 'mae']
                metric_labels = ['R² Score', 'RMSE', 'MAE']
            else:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            
            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
            
            for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = [results[name].get(metric, 0) for name in model_names]
                
                colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(values)))  # type: ignore
                bars = axes[idx].barh(model_names, values, color=colors)
                
                axes[idx].set_xlabel(label, fontsize=11)
                axes[idx].set_title(label, fontsize=12, fontweight='bold')
                axes[idx].grid(axis='x', alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    axes[idx].text(value, bar.get_y() + bar.get_height()/2, 
                                 f'{value:.3f}', ha='left', va='center', fontsize=9)
            
            plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating model comparison plot: {e}")
            return ""
    
    def generate_all_plots(self, model: Any, X_test: Any, y_test: Any,
                          y_pred: Any, model_name: str, problem_type: str,
                          feature_names: Optional[List[str]] = None,
                          y_proba: Optional[Any] = None) -> Dict[str, str]:
        """
        Generate all relevant plots for a model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            y_pred: Predictions
            model_name: Name of the model
            problem_type: Type of problem
            feature_names: Names of features
            y_proba: Predicted probabilities (for classification)
            
        Returns:
            Dictionary of plot types and file paths
        """
        plots = {}
        
        try:
            CLIFormatter.print_info(f"Generating visualizations for {model_name}...")
            
            if problem_type in ['classification', 'binary_classification']:
                # Confusion matrix
                cm_path = self.plot_confusion_matrix(y_test, y_pred, model_name=model_name)
                if cm_path:
                    plots['confusion_matrix'] = cm_path
                
                # ROC curve for binary classification
                if problem_type == 'binary_classification' and y_proba is not None:
                    roc_path = self.plot_roc_curve(y_test, y_proba, model_name=model_name)
                    if roc_path:
                        plots['roc_curve'] = roc_path
            
            else:  # Regression
                # Residual plots
                residual_path = self.plot_residuals(y_test, y_pred, model_name=model_name)
                if residual_path:
                    plots['residuals'] = residual_path
            
            # Learning curve
            lc_path = self.plot_learning_curve(model, X_test, y_test, model_name=model_name)
            if lc_path:
                plots['learning_curve'] = lc_path
            
            # Feature importance
            if feature_names:
                fi_path = self.plot_feature_importance(model, feature_names, model_name=model_name)
                if fi_path:
                    plots['feature_importance'] = fi_path
            
            if plots:
                CLIFormatter.print_success(f"Generated {len(plots)} visualizations")
            
        except Exception as e:
            CLIFormatter.print_error(f"Error generating plots: {str(e)}")
        
        return plots
