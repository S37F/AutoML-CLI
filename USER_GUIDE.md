# 📘 User Guide - How to Use AutoML CLI

## ✅ Yes! New users can easily use this tool with their own CSV files

---

## 🚀 Quick Start for New Users

### Step 1: Prepare Your CSV File
Your CSV file should have:
- ✅ **Headers** in the first row (column names)
- ✅ **At least 50-100 rows** (minimum for training)
- ✅ **Target column** (what you want to predict)
- ✅ **Feature columns** (data used for prediction)

### Step 2: Place Your CSV File
```bash
# Option 1: Place it in the project folder
cp your_data.csv "Dataset-Driven CLI/"

# Option 2: Use absolute path
python automl_cli.py "C:/path/to/your_data.csv"
```

### Step 3: Run the Tool
```bash
# Easiest way (auto mode - no prompts)
python automl_cli.py your_data.csv --auto

# With all features (visualizations + report)
python automl_cli.py your_data.csv --auto --all

# For large datasets (fast mode)
python automl_fast.py your_data.csv --quick
```

---

## 📊 CSV File Requirements

### ✅ Good Example
```csv
age,income,credit_score,loan_amount,employment_years,approved
28,45000,720,150000,3,Yes
35,78000,680,250000,8,No
42,95000,750,300000,15,Yes
...
```

### ❌ Bad Example (Too Small)
```csv
age,income,approved
28,45000,Yes
35,78000,No
(Only 2 rows - NEEDS AT LEAST 50+!)
```

### ✅ Supported Data Types
- **Numeric**: Age, salary, price, score, count
- **Categorical**: Yes/No, categories (will be auto-encoded)
- **Mixed**: Can have both numeric and categorical columns

### ⚠️ Requirements
- **Minimum rows**: 50-100 (recommended 500+ for best results)
- **No special characters** in column names (use underscores: `credit_score` not `credit-score`)
- **Target column**: Should be the column you want to predict

---

## 🎯 Common Use Cases

### 1. House Price Prediction (Regression)
```csv
size_sqft,bedrooms,bathrooms,age_years,location,price
1500,3,2,10,Suburb,250000
2000,4,3,5,City,450000
...
```
**Target**: `price`  
**Command**: `python automl_cli.py houses.csv --auto --all`

### 2. Customer Churn (Classification)
```csv
age,tenure_months,monthly_charges,support_calls,churn
45,24,89.99,2,No
32,6,120.50,5,Yes
...
```
**Target**: `churn`  
**Command**: `python automl_cli.py customers.csv --auto --all`

### 3. Employee Salary Prediction (Regression)
```csv
years_experience,education,department,performance_score,salary
5,Bachelor,Engineering,85,65000
12,Master,Sales,92,95000
...
```
**Target**: `salary`  
**Command**: `python automl_cli.py employees.csv --auto --all`

---

## 🎬 Example Workflow

### New User: Jane wants to predict house prices

```bash
# Jane has house_data.csv with 500 rows
# Columns: size, bedrooms, bathrooms, age, location, price

# Step 1: Copy file to project folder
cp house_data.csv "Dataset-Driven CLI/"

# Step 2: Run with auto mode
cd "Dataset-Driven CLI"
python automl_cli.py house_data.csv --auto --all

# Step 3: Results!
# ✅ 10 models trained in <1 second
# ✅ Best model: Random Forest (R² = 0.94)
# ✅ HTML report: reports/automl_report_20260205_123456.html
# ✅ Visualizations: visualizations/model_comparison.png
# ✅ Model saved: best_model_regression.joblib
```

**Total time: 30 seconds!** ⚡

---

## 💡 Pro Tips for New Users

### 1. Use `--auto` Flag
Skips all prompts and uses smart defaults:
```bash
python automl_cli.py your_data.csv --auto
```

### 2. Get Full Output
Use `--all` to get visualizations, reports, and tuning:
```bash
python automl_cli.py your_data.csv --auto --all
```

### 3. For Large Files
Use fast mode for datasets >100K rows:
```bash
python automl_fast.py big_data.csv --quick
```

### 4. Specify Target Column
If auto-selection is wrong, specify manually:
```bash
# The tool will ask: "Which column do you want to predict?"
# Type: salary (or whatever your target is)
python automl_cli.py your_data.csv
```

---

## 🔧 Troubleshooting

### ❌ "No results to display"
**Cause**: Dataset too small (< 50 rows)  
**Fix**: Use at least 100 rows, recommended 500+

### ❌ "Column not found"
**Cause**: Wrong column name or typo  
**Fix**: Check your CSV headers match exactly

### ❌ "All models failed"
**Cause**: Dataset too small or target column has issues  
**Fix**: 
- Use at least 100 rows
- Ensure target column has variation (not all same value)
- Check for extreme class imbalance (e.g., 99% one class)

### ❌ "Training too slow"
**Cause**: Large dataset  
**Fix**: Use `--quick` flag or `automl_fast.py`

---

## 📈 Expected Results

### Small Dataset (100-1,000 rows)
- Training time: <5 seconds
- Memory: <50MB
- Accuracy: 70-90% (depends on data quality)

### Medium Dataset (1,000-100,000 rows)
- Training time: 10-60 seconds
- Memory: <500MB
- Accuracy: 80-95%

### Large Dataset (100,000+ rows)
- Auto-sampled to 100,000 rows
- Training time: 1-3 minutes
- Memory: <1GB
- Accuracy: 85-98%

---

## 🎯 What You Get

After running the command, you'll have:

1. **Console Output**
   - Dataset analysis (rows, columns, missing values)
   - Problem type detection (regression/classification)
   - Model comparison table
   - Best model with metrics

2. **Files Generated**
   - `visualizations/` → Beautiful charts (PNG, 300 DPI)
   - `reports/` → Professional HTML report
   - `best_model_*.joblib` → Trained model ready for deployment

3. **Model Metrics**
   - **Regression**: R², RMSE, MAE, Cross-validation score
   - **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC

---

## ✅ Success Checklist

Before running, ensure:
- [ ] CSV file has headers
- [ ] At least 100 rows (500+ recommended)
- [ ] Target column is clear (what to predict)
- [ ] No special characters in column names
- [ ] File is accessible (correct path)

---

## 🚀 Real-World Example

### Sarah's Marketing Campaign

**Scenario**: Sarah has customer data and wants to predict who will buy.

**Her CSV** (`customers.csv`):
```csv
age,income,website_visits,email_opens,past_purchases,will_buy
34,65000,12,8,3,Yes
28,45000,5,2,0,No
42,85000,20,15,7,Yes
... (500 more rows)
```

**Command**:
```bash
python automl_cli.py customers.csv --auto --all
```

**Results** (30 seconds later):
- ✅ Binary classification detected
- ✅ 10 models trained
- ✅ Best: Random Forest (F1 = 0.87, 87% accuracy)
- ✅ HTML report with ROC curve and confusion matrix
- ✅ Model saved for deployment

**Sarah deploys the model** and increases conversion by 23%! 🎯

---

## 📧 Support

If you encounter issues:
1. Check CSV has 100+ rows
2. Verify column names (no special chars)
3. Use `--auto` flag to skip prompts
4. Check [DEMO.md](DEMO.md) for more examples

---

## 🎓 Summary

### ✅ **Yes, new users can easily use this tool!**

**3 Simple Steps:**
1. Prepare CSV file (100+ rows, clear headers)
2. Run: `python automl_cli.py your_file.csv --auto --all`
3. Get results: trained model + visualizations + report

**No ML knowledge required!** The tool handles:
- ✅ Problem detection (regression vs classification)
- ✅ Data preprocessing (missing values, encoding, scaling)
- ✅ Model selection (chooses best algorithms)
- ✅ Training & evaluation (parallel processing)
- ✅ Hyperparameter tuning (optional with `--tune`)
- ✅ Professional outputs (reports, charts, saved models)

**Perfect for:**
- Data scientists (rapid prototyping)
- Business analysts (quick insights)
- Students (learning ML workflows)
- Anyone with CSV data who wants predictions!

---

**🎯 Bottom Line**: Drop in your CSV, run one command, get trained ML model in 30 seconds!
