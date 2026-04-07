# 🤖 AutoML CLI - Intelligent Machine Learning Automation

> **Production-ready AutoML tool that transforms raw CSV data into trained ML models with a single command**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)

An enterprise-grade, intelligent AutoML CLI tool that automatically analyzes datasets, selects optimal algorithms, trains models with hyperparameter tuning, and generates professional visualizations and reports.

## 🎯 Key Features

### 🚀 Intelligent Automation
- **Auto-Detection**: Automatically identifies regression, binary, or multi-class classification
- **Smart Sampling**: Intelligently handles datasets from 100 rows to 1M+ rows
- **Model Selection**: Chooses optimal algorithms based on dataset characteristics
- **Parallel Processing**: 5-10x faster training using multi-core CPUs

### 🤖 Advanced ML Capabilities
- **13+ Algorithms**: Linear/Ridge/Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM, KNN, SVM, Naive Bayes
- **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV with model-specific parameter grids
- **Auto-Preprocessing**: Handles missing values, categorical encoding, feature scaling
- **Cross-Validation**: 5-fold CV for robust model evaluation

### 📊 Professional Outputs
- **Visualizations**: Confusion matrices, ROC curves, residual plots (300 DPI PNG)
- **HTML Reports**: Beautiful gradient-styled reports with embedded charts
- **Model Persistence**: Save/load trained models (.joblib format)
- **Performance Metrics**: Comprehensive evaluation (R², MAE, RMSE, Accuracy, F1, ROC-AUC)

### 🎨 User Experience
- **Beautiful CLI**: Color-coded output with emojis and progress bars
- **Interactive**: Smart prompts with auto-suggestions
- **Fast Mode**: Optimized for large datasets (>100K rows)
- **Error Handling**: Graceful degradation and helpful error messages

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/automl-cli.git
cd automl-cli

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data (Optional)
```bash
python generate_sample_data.py
```
Creates 3 sample datasets: house_prices.csv (regression), loan_approval.csv (binary classification), iris_classification.csv (multi-class)

### 3. Run AutoML
```bash
# Quick demo (30 seconds)
python automl_cli.py data/house_prices.csv

# Full pipeline with all features (2-3 minutes)
python automl_cli.py data/house_prices.csv --all

# Large dataset mode (auto-samples to 100K rows)
python automl_fast.py large_dataset.csv --quick
```

### 4. View Results
- **Console**: Colored output with model comparison table
- **Visualizations**: `visualizations/*.png` (confusion matrix, residual plots)
- **Reports**: `reports/*.html` (professional HTML report)
- **Models**: `best_model_*.joblib` (trained model ready for deployment)

## 📖 Usage Examples

### Fast Mode (Recommended for Large Datasets)
```bash
# Auto-sample + skip slow models
python automl_fast.py instagram_usage_lifestyle.csv --quick

# Custom sampling
python automl_fast.py large_data.csv --sample 50000 --quick

# With visualizations and model saving
python automl_fast.py data.csv --quick --visualize --save-model

# Complete workflow
python automl_fast.py data.csv --quick --all
```

### Regular Mode
```bash
# Basic run
python automl_cli.py data/dataset.csv

# With hyperparameter tuning
python automl_cli.py data/dataset.csv --tune

# All features
python automl_cli.py data/dataset.csv --all
```

## 🎯 Command Line Arguments

### automl_fast.py (For Large Datasets)
```
--sample N          Use only N random rows (default: auto 100K)
--quick             Skip slow models (KNN, SVM)
--tune              Enable hyperparameter tuning
--visualize         Generate visualizations
--report            Generate HTML report
--save-model        Save best model as .joblib
--all               Enable all features
--test-size FLOAT   Test split ratio (default: 0.15)
```

### automl_cli.py (Standard)
```
--tune              Enable hyperparameter tuning
--visualize         Generate visualizations  
--report            Generate HTML report
--save-model        Save best model
--all               Enable all features
--test-size FLOAT   Test split ratio (default: 0.2)
--no-parallel       Disable parallel training
```

## 📊 Supported Algorithms

### Regression (10 models)
- Linear/Lasso/Ridge Regression
- Random Forest, Gradient Boosting
- XGBoost, LightGBM
- Decision Tree, KNN, SVR

### Classification (10 models)
- Logistic Regression, Naive Bayes
- Random Forest, Gradient Boosting
- XGBoost, LightGBM
- Decision Tree, KNN, SVM

## 📁 Output Files

```
Dataset-Driven CLI/
├── visualizations/           # PNG plots (300 DPI)
│   ├── confusion_matrix_*.png
│   └── residuals_*.png
├── reports/                  # HTML reports
│   └── automl_report_*.html
└── best_model_*.joblib      # Saved models
```

## 💡 Tips for Large Datasets

1. **Use automl_fast.py** - Optimized for datasets >100K rows
2. **Add --quick flag** - Skips KNN/SVM which are slow on large data
3. **Customize sampling** - Use `--sample 50000` for your preferred size
4. **Skip tuning initially** - Add `--tune` only after selecting best approach

## 🔧 Performance

| Dataset Size | Mode | Time | Command |
|-------------|------|------|---------|
| <10K rows | Regular | 1-2 min | `python automl_cli.py data.csv` |
| 10K-100K | Fast | 2-3 min | `python automl_fast.py data.csv --quick` |
| 100K-1M+ | Fast+Sample | 2-4 min | `python automl_fast.py data.csv --quick --sample 100000` |

## 📚 Dependencies

Core: `pandas`, `numpy`, `scikit-learn`, `joblib`
Advanced: `xgboost`, `lightgbm` (optional)
Visualization: `matplotlib`, `seaborn`
CLI: `colorama`, `tqdm`, `tabulate`, `rich`
Reports: `jinja2`

## 🎓 Use Cases

- Quick ML prototyping
- Kaggle competitions
- Business analytics
- Academic research
- Production model selection

## 💾 Model Loading

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load('best_model_20240204_143022.joblib')

# Make predictions
new_data = pd.DataFrame({...})
predictions = model.predict(new_data)
```

## 🐛 Troubleshooting

**Large dataset too slow?**
- Use `python automl_fast.py` instead
- Add `--quick` flag
- Reduce sample size: `--sample 50000`

**Out of memory?**
- Reduce `--sample` size
- Close other applications
- Use `--quick` to skip memory-intensive models

**XGBoost/LightGBM not found?**
- Install: `pip install xgboost lightgbm`
- Or continue without them (tool will skip gracefully)

## 🎓 Skills Demonstrated

This project showcases:
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, model selection, hyperparameter tuning
- **Python Development**: OOP, type hints, error handling, modular architecture
- **Data Processing**: pandas, NumPy, data cleaning, feature engineering
- **CLI Development**: argparse, colored output, progress tracking, user interaction
- **Visualization**: Matplotlib, Seaborn, professional chart generation
- **Software Engineering**: Clean code, documentation, testing, version control
- **Performance Optimization**: Parallel processing, efficient algorithms, memory management

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- Portfolio: [your-portfolio.com](https://your-portfolio.com)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

## 🌟 Acknowledgments

Built with:
- [scikit-learn](https://scikit-learn.org/) - Machine learning framework
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library
- [LightGBM](https://lightgbm.readthedocs.io/) - Fast gradient boosting
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualization

---

**⭐ Star this repo if you found it helpful!**

*Made with ❤️ for the ML community*
