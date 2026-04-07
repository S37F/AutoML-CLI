"""
Report Generator Module
Generate comprehensive HTML reports with visualizations
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import base64
from jinja2 import Template


class ReportGenerator:
    """Generate HTML reports for ML experiments"""
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for embedding in HTML"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        except:
            return ""
    
    def _get_html_template(self) -> str:
        """Get HTML template for the report"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Report - {{ experiment_name }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
            border-left: 4px solid #667eea;
            padding-left: 20px;
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .info-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .info-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .info-card p {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        tr:hover {
            background-color: #e9ecef;
        }
        
        .best-model {
            background-color: #d4edda !important;
            font-weight: bold;
        }
        
        .best-model td:first-child::before {
            content: "⭐ ";
        }
        
        .metric-good { color: #28a745; font-weight: bold; }
        .metric-medium { color: #ffc107; font-weight: bold; }
        .metric-poor { color: #dc3545; font-weight: bold; }
        
        .visualization {
            margin: 30px 0;
            text-align: center;
        }
        
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin: 10px 0;
        }
        
        .visualization h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .best-model-highlight {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        
        .best-model-highlight h3 {
            font-size: 1.8em;
            margin-bottom: 15px;
        }
        
        .best-model-highlight p {
            font-size: 1.1em;
            margin: 5px 0;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .parameter-list {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .parameter-list li {
            margin: 5px 0;
            list-style-position: inside;
        }
        
        @media print {
            body { background: white; padding: 0; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AutoML Experiment Report</h1>
            <p>{{ experiment_name }}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated on {{ timestamp }}</p>
        </div>
        
        <div class="content">
            <!-- Dataset Summary -->
            <div class="section">
                <h2>📊 Dataset Summary</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Dataset Path</h3>
                        <p style="font-size: 1em; word-break: break-all;">{{ dataset_info.path }}</p>
                    </div>
                    <div class="info-card">
                        <h3>Total Samples</h3>
                        <p>{{ dataset_info.total_samples }}</p>
                    </div>
                    <div class="info-card">
                        <h3>Total Features</h3>
                        <p>{{ dataset_info.total_features }}</p>
                    </div>
                    <div class="info-card">
                        <h3>Problem Type</h3>
                        <p style="font-size: 1.2em;">{{ dataset_info.problem_type }}</p>
                    </div>
                    <div class="info-card">
                        <h3>Target Column</h3>
                        <p style="font-size: 1.2em;">{{ dataset_info.target_column }}</p>
                    </div>
                    <div class="info-card">
                        <h3>Missing Values</h3>
                        <p>{{ dataset_info.missing_values }}</p>
                    </div>
                </div>
            </div>
            
            <!-- Model Performance Comparison -->
            <div class="section">
                <h2>🏆 Model Performance Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model Name</th>
                            {% for metric in metrics %}
                            <th>{{ metric }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for model_name, model_results in results.items() %}
                        <tr {% if model_name == best_model_name %}class="best-model"{% endif %}>
                            <td>{{ model_name }}</td>
                            {% for metric in metrics %}
                            <td>{{ "%.4f"|format(model_results.get(metric, 0)) }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <!-- Best Model -->
            <div class="section">
                <div class="best-model-highlight">
                    <h3>🥇 Best Model: {{ best_model_name }}</h3>
                    {% for metric, value in best_model_metrics.items() %}
                    <p><strong>{{ metric }}:</strong> {{ "%.4f"|format(value) }}</p>
                    {% endfor %}
                    
                    {% if tuned_params %}
                    <h4 style="margin-top: 20px;">Optimized Parameters:</h4>
                    <ul class="parameter-list">
                        {% for param, value in tuned_params.items() %}
                        <li><strong>{{ param }}:</strong> {{ value }}</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                </div>
            </div>
            
            <!-- Visualizations -->
            {% if visualizations %}
            <div class="section">
                <h2>📈 Visualizations</h2>
                
                {% if visualizations.model_comparison %}
                <div class="visualization">
                    <h3>Model Comparison</h3>
                    <img src="data:image/png;base64,{{ visualizations.model_comparison }}" alt="Model Comparison">
                </div>
                {% endif %}
                
                {% if visualizations.confusion_matrix %}
                <div class="visualization">
                    <h3>Confusion Matrix</h3>
                    <img src="data:image/png;base64,{{ visualizations.confusion_matrix }}" alt="Confusion Matrix">
                </div>
                {% endif %}
                
                {% if visualizations.roc_curve %}
                <div class="visualization">
                    <h3>ROC Curve</h3>
                    <img src="data:image/png;base64,{{ visualizations.roc_curve }}" alt="ROC Curve">
                </div>
                {% endif %}
                
                {% if visualizations.residuals %}
                <div class="visualization">
                    <h3>Residual Analysis</h3>
                    <img src="data:image/png;base64,{{ visualizations.residuals }}" alt="Residual Plot">
                </div>
                {% endif %}
                
                {% if visualizations.learning_curve %}
                <div class="visualization">
                    <h3>Learning Curve</h3>
                    <img src="data:image/png;base64,{{ visualizations.learning_curve }}" alt="Learning Curve">
                </div>
                {% endif %}
                
                {% if visualizations.feature_importance %}
                <div class="visualization">
                    <h3>Feature Importance</h3>
                    <img src="data:image/png;base64,{{ visualizations.feature_importance }}" alt="Feature Importance">
                </div>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Experiment Configuration -->
            <div class="section">
                <h2>⚙️ Experiment Configuration</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Test Split</h3>
                        <p>{{ "%.0f"|format(config.test_size * 100) }}%</p>
                    </div>
                    <div class="info-card">
                        <h3>Models Trained</h3>
                        <p>{{ config.models_trained }}</p>
                    </div>
                    <div class="info-card">
                        <h3>Hyperparameter Tuning</h3>
                        <p>{{ config.tuning_enabled }}</p>
                    </div>
                    <div class="info-card">
                        <h3>Interpretability Priority</h3>
                        <p>{{ config.interpretability }}</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by AutoML CLI - Dataset-Driven Automated Machine Learning Tool</p>
            <p>Powered by scikit-learn, XGBoost, LightGBM</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_report(self, 
                       experiment_name: str,
                       dataset_info: Dict,
                       results: Dict[str, Dict],
                       best_model_name: str,
                       best_model_metrics: Dict,
                       problem_type: str,
                       config: Dict,
                       visualizations: Optional[Dict[str, str]] = None,
                       tuned_params: Optional[Dict] = None) -> str:
        """
        Generate comprehensive HTML report
        
        Args:
            experiment_name: Name of the experiment
            dataset_info: Information about the dataset
            results: Results for all models
            best_model_name: Name of the best model
            best_model_metrics: Metrics for the best model
            problem_type: Type of ML problem
            config: Experiment configuration
            visualizations: Dictionary of visualization file paths
            tuned_params: Hyperparameters from tuning
            
        Returns:
            Path to generated report
        """
        # Convert images to base64
        viz_base64 = {}
        if visualizations:
            for viz_type, viz_path in visualizations.items():
                if viz_path and os.path.exists(viz_path):
                    viz_base64[viz_type] = self._image_to_base64(viz_path)
        
        # Determine metrics based on problem type
        if problem_type == 'regression':
            metrics = ['RMSE', 'MAE', 'R²', 'CV Score']
            metric_keys = ['rmse', 'mae', 'r2', 'cv_score_mean']
        else:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'CV Score']
            metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'cv_score_mean']
        
        # Prepare template data
        template_data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': dataset_info,
            'results': results,
            'best_model_name': best_model_name,
            'best_model_metrics': best_model_metrics,
            'metrics': metrics,
            'metric_keys': metric_keys,
            'problem_type': problem_type.replace('_', ' ').title(),
            'config': config,
            'visualizations': viz_base64,
            'tuned_params': tuned_params or {}
        }
        
        # Render template
        template = Template(self._get_html_template())
        html_content = template.render(**template_data)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"automl_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
