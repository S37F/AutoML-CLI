# 📹 AutoML CLI Demo

## 🎬 Quick Demo (30 seconds)

```bash
python automl_cli.py data/house_prices.csv
```

**What happens:**
1. ✅ Loads 500-row dataset
2. ✅ Auto-detects regression problem
3. ✅ Selects 10 optimal algorithms
4. ✅ Trains models in parallel (~10 seconds)
5. ✅ Displays performance comparison
6. ✅ Shows best model (typically Random Forest with R² ~0.92)

## 📊 Sample Output

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        🤖 DATASET-DRIVEN AUTOMATED ML CLI TOOL 🤖              ║
║                                                                  ║
║              Intelligent • Explainable • Efficient               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

[STEP 1/7] 📊 Loading and Analyzing Dataset
✓ Dataset loaded: 500 rows × 11 columns

──────────────────────────────────────────────────────────────────
  DATASET ANALYSIS
──────────────────────────────────────────────────────────────────
Dataset Shape: 500 rows × 11 columns

📊 Numeric Columns (9)
  • Age_Years (unique: 42, missing: 0)
  • Square_Feet (unique: 489, missing: 0)
  • Bedrooms (unique: 4, missing: 0)
  ...

✓ No missing values detected!

[STEP 2/7] 🎯 Target Column Selection
Suggested targets: Price, Square_Feet, Income
Target column: Price

[STEP 3/7] 🔍 Problem Type Detection
✓ Regression detected
  • Target has 487 unique values (97% of data)
  • Numeric target suggests regression

[STEP 7/7] 🚀 Training Models (Parallel)
Training 10 models in parallel...
[████████████████████] 100% | ETA: 0s

──────────────────────────────────────────────────────────────────
  MODEL PERFORMANCE COMPARISON
──────────────────────────────────────────────────────────────────

Model                          R²      MAE        RMSE       Time
────────────────────────────────────────────────────────────────
Random Forest Regressor      0.9234   15234.21   21543.87   0.45s
XGBoost Regressor            0.9187   16012.43   22876.12   0.32s
Gradient Boosting            0.9156   16543.11   23234.55   0.28s
LightGBM Regressor           0.9123   17234.32   23876.43   0.21s
Ridge Regression             0.8765   21234.54   28765.32   0.03s
Linear Regression            0.8756   21345.67   28876.21   0.02s
...

══════════════════════════════════════════════════════════════════
  🏆 BEST MODEL FOUND
══════════════════════════════════════════════════════════════════
  Model: Random Forest Regressor
  Score: 0.9234
  MAE: $15,234.21
  RMSE: $21,543.87
══════════════════════════════════════════════════════════════════

🎉 AutoML Complete!
```

## 🎯 Use Cases

### 1. Regression Example (House Prices)
```bash
python automl_cli.py data/house_prices.csv --all
```
**Output:**
- Best Model: Random Forest (R² = 0.92)
- Visualizations: Residual plot, learning curve
- Report: `reports/automl_report_*.html`

### 2. Binary Classification (Loan Approval)
```bash
python automl_cli.py data/loan_approval.csv --all
```
**Output:**
- Best Model: XGBoost (Accuracy = 95.4%, ROC-AUC = 0.987)
- Visualizations: Confusion matrix, ROC curve
- Report: Performance metrics for all models

### 3. Multi-class Classification (Iris)
```bash
python automl_cli.py data/iris_classification.csv --all
```
**Output:**
- Best Model: Random Forest (Accuracy = 96.7%)
- Visualizations: Confusion matrix
- Report: Class-wise precision/recall

## 📈 Full Pipeline Demo (2-3 minutes)

```bash
python automl_cli.py data/house_prices.csv --all
```

**Features demonstrated:**
1. ✅ **Data Analysis**: Summary statistics, missing value detection
2. ✅ **Smart Selection**: Auto-suggests target column
3. ✅ **Model Training**: 10 models trained in parallel
4. ✅ **Hyperparameter Tuning**: RandomizedSearchCV optimization
5. ✅ **Visualizations**: 
   - Residual plot (regression)
   - Learning curve
   - Feature importance
   - Model comparison chart
6. ✅ **HTML Report**: Beautiful report with all results
7. ✅ **Model Saving**: `best_model_*.joblib` for deployment

## 🚀 Large Dataset Demo

```bash
# 1.5M rows → auto-samples to 100K
python automl_fast.py instagram_usage_lifestyle.csv --quick
```

**Performance:**
- Original: 1,547,896 rows
- Sampled: 100,000 rows (automatic)
- Time: ~2-3 minutes (vs 30+ minutes for full dataset)
- Models: 7 fast models (skips KNN/SVM)

## 📊 Visualizations Generated

### Regression
1. **Residual Plot**: Predicted vs Actual residuals
2. **Learning Curve**: Training vs validation score
3. **Feature Importance**: Top features ranked
4. **Model Comparison**: Bar chart of all model R² scores

### Classification
1. **Confusion Matrix**: Heatmap with accuracy breakdown
2. **ROC Curve**: True/False positive rate (binary only)
3. **Feature Importance**: Top predictive features
4. **Model Comparison**: Accuracy/F1 scores

## 💼 Business Use Case Example

### Scenario: Predict Customer Churn
```bash
python automl_cli.py customer_churn.csv --all
```

**Deliverables for stakeholders:**
1. **Executive Summary**: Best model accuracy: 94.2%
2. **Technical Report**: `reports/automl_report_*.html`
   - All model comparisons
   - Feature importance analysis
   - Performance metrics
3. **Deployment Ready**: `best_model_*.joblib`
4. **Visualizations**: ROC curve, confusion matrix for presentation

## 🎓 Educational Demo

Perfect for learning ML workflows:

```bash
# 1. Start simple
python automl_cli.py data/iris_classification.csv

# 2. Add tuning
python automl_cli.py data/iris_classification.csv --tune

# 3. Full analysis
python automl_cli.py data/iris_classification.csv --all
```

Shows progression from basic to advanced ML pipeline.

## 🔄 Model Deployment Example

```python
# After running: python automl_cli.py data/house_prices.csv --save-model

import joblib
import pandas as pd

# Load trained model
model = joblib.load('best_model_20260204_143022.joblib')

# Predict on new data
new_house = pd.DataFrame({
    'Square_Feet': [2500],
    'Bedrooms': [4],
    'Age_Years': [10],
    # ... other features
})

predicted_price = model.predict(new_house)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

## 📸 Screenshot Highlights

### Terminal Output
- ✅ Colored, emoji-enhanced output
- ✅ Progress bars for training
- ✅ Beautiful table formatting
- ✅ Clear step-by-step workflow

### HTML Report
- ✅ Professional gradient design
- ✅ Responsive layout
- ✅ Embedded visualizations
- ✅ Comprehensive metrics

### Visualizations
- ✅ 300 DPI publication-quality
- ✅ Color-coded for clarity
- ✅ Professional styling
- ✅ Ready for presentations

---

**Ready to impress recruiters? This demo shows:**
- ✅ End-to-end ML pipeline
- ✅ Production-ready code
- ✅ Professional documentation
- ✅ User-friendly interface
- ✅ Scalability (100 rows to 1M+ rows)
- ✅ Real-world applicability
