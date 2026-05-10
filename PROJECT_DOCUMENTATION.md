# AutoML-CLI — Comprehensive Project Documentation

> **Note:** This project is a **local Python CLI for AutoML**, not a web API or networked backend. There are no HTTP routes, JWT/OAuth, or application databases.

---

## 1. Tech Stack

### Runtime & language

| Item | Version / notes |
|------|-----------------|
| **Python** | README states 3.8+ (no `pyproject.toml` or pinned interpreter in repo) |

### Core ML & data (from `requirements.txt`)

| Package | Constraint |
|---------|------------|
| pandas | >= 1.5.0 |
| numpy | >= 1.23.0 |
| scikit-learn | >= 1.2.0 |
| joblib | >= 1.2.0 |
| xgboost | >= 1.7.0 |
| lightgbm | >= 3.3.0 |

### Visualization & reporting

| Package | Constraint |
|---------|------------|
| matplotlib | >= 3.6.0 |
| seaborn | >= 0.12.0 |
| jinja2 | >= 3.1.0 |
| plotly | >= 5.14.0 |
| kaleido | >= 0.2.1 |

**Note:** `plotly` and `kaleido` are listed in `requirements.txt` but **no `.py` file imports them**. Plotting is implemented with **matplotlib** and **seaborn** in `visualizer.py`.

### CLI & UX

| Package | Constraint |
|---------|------------|
| colorama | >= 0.4.6 |
| tqdm | >= 4.65.0 |
| tabulate | >= 0.9.0 |
| rich | >= 13.0.0 |

**Note:** `rich` is in `requirements.txt` but **not imported** in the Python sources; CLI styling uses `colorama` and `tabulate` via `cli_utils.py`.

### Testing / tooling

- **`test_portfolio.py`**: integration-style checks using `subprocess.run` (not pytest/unittest).
- **DevOps:** No Docker, GitHub Actions, or CI config found in the explored tree; `.gitignore` covers Python artifacts, venvs, `*.joblib`, etc.

---

## 2. Backend Concepts & Key Methods

This codebase has **no HTTP server**. Below, “backend” is interpreted as **core application and ML pipeline concepts** actually implemented.

### 1) Command-line interface (argparse)

- **What:** Standard-library parsing of script arguments and flags.
- **Why:** Drive the full AutoML flow from the shell without a GUI or API.
- **How:** `automl_cli.py` defines `ArgumentParser`, positional `dataset`, and flags like `--tune`, `--visualize`, `--report`, `--all`, `--sample`, `--quick`, `--auto`, `--no-parallel`, `--test-size`; `args = parser.parse_args()`, then `main()` orchestrates steps.
- **Where:** `automl_cli.py` — `main()`.

### 2) Interactive user input (target column, interpretability)

- **What:** Prompted input with optional validation against allowed options.
- **Why:** Let the user pick the prediction column and interpretability preference when not using `--auto`.
- **How:** `get_user_input()` / `get_yes_no_input()` call `CLIFormatter.get_input()`; `--auto` skips prompts and uses `analyzer.suggest_target_columns()[0]` and `interpretability = False`.
- **Where:** `automl_cli.py` — `get_user_input`, `get_yes_no_input`, `main`.

### 3) Dataset analysis & problem-type detection

- **What:** Load CSV into pandas, compute stats, infer regression vs classification from the target column.
- **Why:** Automate choosing the right metrics and model families.
- **How:** `DatasetAnalyzer.load_dataset()` uses `pd.read_csv`; `analyze_dataset()` builds shape, dtypes, missing counts, numeric/categorical splits; `identify_problem_type()` uses target `dtype`, `nunique()`, and thresholds (e.g. 2 → binary, ≤10 unique numeric → multi-class, else regression).
- **Where:** `dataset_analyzer.py` — class `DatasetAnalyzer`.

### 4) Model portfolio selection (heuristics)

- **What:** Build a dict of sklearn (and optionally XGBoost/LightGBM) estimators by problem type, interpretability, and dataset size.
- **Why:** Balance accuracy vs speed and interpretability (e.g. skip KNN/SVR on large data when interpretability is off).
- **How:** `ModelSelector.select_models()` dispatches to `_select_regression_models` / `_select_classification_models`; optional imports set `XGBOOST_AVAILABLE` / `LIGHTGBM_AVAILABLE`.
- **Where:** `model_selector.py` — `ModelSelector`.

### 5) Data preprocessing (imputation, encoding, scaling)

- **What:** sklearn `SimpleImputer`, `LabelEncoder`, `StandardScaler` on features; label-encode categorical target when needed.
- **Why:** Prepare mixed-type tabular data for sklearn pipelines.
- **How:** `DataPreprocessor.fit_transform()` sets `numeric_columns` / `categorical_columns`, fits imputers, encodes categoricals, scales numerics.
- **Where:** `preprocessor.py` — `DataPreprocessor`.

### 6) Train / test split

- **What:** Holdout split for evaluation.
- **Why:** Report metrics on unseen data.
- **How:** `train_test_split(X, y, test_size=..., random_state=42)` in `ModelTrainer.train_and_evaluate`.
- **Where:** `model_trainer.py` — `train_and_evaluate`.

### 7) Parallel model training (joblib threading)

- **What:** Train multiple models concurrently when the portfolio is large enough.
- **Why:** Reduce wall-clock time on multi-core CPUs.
- **How:** If `parallel and len(models) > 2`, `Parallel(n_jobs=-1, backend='threading')` with `delayed(self._train_single_model)(...)`; else sequential `tqdm` loop.
- **Where:** `model_trainer.py` — `train_and_evaluate`, `_train_single_model`.

### 8) Cross-validation on training set

- **What:** `cross_val_score` on training data for each fitted model.
- **Why:** Extra stability signal beyond the single holdout split.
- **How:** `cross_val_score(..., cv=min(5, len(X_train)//10), scoring='r2' or 'accuracy', n_jobs=1)` inside `_train_single_model`.
- **Where:** `model_trainer.py` — `_train_single_model`.

### 9) Metrics & best-model selection

- **What:** Regression: RMSE, MAE, R²; classification: accuracy, precision, recall, F1 (binary vs weighted average).
- **Why:** Comparable leaderboard and automatic “best” pick.
- **How:** `_evaluate_*` builds dicts; `_select_best_model` uses `'r2'` for regression and `'f1'` for classification when `metric_priority == 'auto'`.
- **Where:** `model_trainer.py` — `_evaluate_regression`, `_evaluate_classification`, `_select_best_model`.

### 10) Hyperparameter tuning

- **What:** `RandomizedSearchCV` (default) or `GridSearchCV` over model-specific grids.
- **Why:** Improve the winning model after the initial bake-off.
- **How:** `HyperparameterTuner.tune_model()` builds search from `param_grids`; on Windows `n_jobs = 1` to avoid multiprocessing issues (`IS_WINDOWS`).
- **Where:** `hyperparameter_tuner.py` — `HyperparameterTuner`, `tune_model`.

### 11) Optional dependency pattern

- **What:** Try/import optional libraries; continue without them.
- **Why:** Install can succeed without XGBoost/LightGBM; models that need them are omitted.
- **How:** `try: import xgboost` / `lightgbm` in `model_selector.py` and `hyperparameter_tuner.py`.
- **Where:** Top of those modules and conditional model dict entries.

### 12) “Fast” entrypoint (argument injection)

- **What:** Thin wrapper that appends default CLI flags before delegating.
- **Why:** Large data: default `--sample 100000` and `--quick` without typing them every time.
- **How:** `automl_fast.py` mutates `sys.argv` then `from automl_cli import main; main()`.
- **Where:** `automl_fast.py`.

### 13) Static visualization artifacts

- **What:** Matplotlib `Agg` backend, PNG files to disk.
- **Why:** Offline charts for reports and folders.
- **How:** `ModelVisualizer` methods (`plot_confusion_matrix`, `plot_roc_curve`, `plot_residuals`, `plot_learning_curve`, `plot_feature_importance`, `plot_model_comparison`, `generate_all_plots`).
- **Where:** `visualizer.py`.

### 14) HTML report generation

- **What:** Jinja2 template rendered to timestamped HTML; images embedded as base64.
- **Why:** Single shareable artifact with tables and figures.
- **How:** `ReportGenerator.generate_report()` fills `Template(self._get_html_template()).render(...)`, writes `reports/automl_report_<timestamp>.html`.
- **Where:** `report_generator.py` — `ReportGenerator`.

### 15) Model persistence

- **What:** Serialize best estimator with joblib.
- **Why:** Deploy or reload later (`README` shows `joblib.load`).
- **How:** `joblib.dump(self.best_model, filepath)` in `ModelTrainer.save_model`; invoked from `main` as `best_model_{problem_type}.joblib`.
- **Where:** `model_trainer.py` — `save_model`; call sites in `automl_cli.py`.

### Not present (would be “generic backend” answers elsewhere)

- **JWT, OAuth, sessions, API keys:** not implemented.
- **WebSockets:** none.
- **HTTP rate limiting, CORS, Helmet:** N/A (no web server).
- **Message queues / webhooks:** none.
- **Server-side caching (Redis, etc.):** none.

---

## 3. API Inventory

There are **no HTTP REST/GraphQL endpoints**. The public “interfaces” are **CLI entry points** and their arguments.

### Script: `automl_cli.py` (primary)

| Invocation element | Description | “Auth” | Arguments / input | Output |
|--------------------|-------------|--------|-------------------|--------|
| Positional `dataset` | Path to CSV | N/A (local file) | String path | Loaded into `DatasetAnalyzer` |
| `--save-model` | Save best model | N/A | Flag | `best_model_<problem_type>.joblib` |
| `--test-size` | Holdout fraction | N/A | Float, default `0.2` | Affects `train_test_split` |
| `--no-parallel` | Sequential training | N/A | Flag | Disables `Parallel` path |
| `--tune` | Tune best model | N/A | Flag | `HyperparameterTuner.tune_model` |
| `--visualize` | PNG charts | N/A | Flag | Files under `visualizations/` |
| `--report` | HTML report (+ save model if best exists) | N/A | Flag | `reports/automl_report_*.html` |
| `--all` | Enables tune, visualize, report, save | N/A | Flag | Sets multiple flags in `main` |
| `--sample` | Row subsample | N/A | Integer | `DataFrame.sample` |
| `--quick` | Skip slow models (KNN/SVM) when rows > 5000 | N/A | Flag | Filters `models` dict |
| `--auto` | No prompts | N/A | Flag | Auto target + no interpretability question |

**Interactive stdin (when not `--auto`):** user provides target column name and y/n for interpretability — not flags.

### Script: `automl_fast.py`

| Element | Description |
|---------|-------------|
| Effect | Injects `--sample 100000` and `--quick` into `sys.argv` if missing, then runs `automl_cli.main` |

### Script: `generate_sample_data.py`

| Element | Description |
|---------|-------------|
| Running the module | Writes `data/house_prices.csv`, `data/loan_approval.csv`, `data/iris_classification.csv` |

### Script: `test_portfolio.py`

| Element | Description |
|---------|-------------|
| Subprocess commands | e.g. `python automl_cli.py data/house_prices.csv --auto --all` with piped stdin |

---

## 4. Architecture Diagram (Text-based)

```
[User / Terminal]
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Entry points                                                   │
│  automl_cli.py :: main()                                      │
│  automl_fast.py :: inject --sample/--quick → main()           │
│  generate_sample_data.py :: write sample CSVs                  │
│  test_portfolio.py :: subprocess smoke tests                   │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ Orchestration (automl_cli.main)                                │
│  argparse → steps: analyze → target → problem type →          │
│  ModelSelector → DataPreprocessor → ModelTrainer →            │
│  [optional] HyperparameterTuner → ModelVisualizer →           │
│  [optional] ReportGenerator / save_model                      │
└─────────────┬─────────────────────────────┬─────────────────┘
              │                             │
              ▼                             ▼
   ┌────────────────────┐         ┌────────────────────┐
   │ dataset_analyzer   │         │ model_selector     │
   │ (CSV → DataFrame)  │         │ (sklearn + opt.    │
   │                    │         │  xgb/lgb)          │
   └─────────┬──────────┘         └─────────┬───────────┘
             │                            │
             ▼                            ▼
   ┌────────────────────┐         ┌────────────────────┐
   │ preprocessor       │         │ model_trainer      │
   │ impute/encode/     │         │ train_test_split,  │
   │ scale              │         │ Parallel/tqdm, CV, │
   └─────────┬──────────┘         │ metrics, best pick │
             │                   └─────────┬───────────┘
             │                             │
             │                             ▼
             │                   ┌────────────────────┐
             │                   │ hyperparameter_    │
             │                   │ tuner (optional)   │
             │                   └─────────┬──────────┘
             │                             │
             ▼                             ▼
   ┌─────────────────────────────────────────────────────┐
   │ Filesystem outputs (no application database)       │
   │  • visualizations/*.png                             │
   │  • reports/*.html (Jinja2 + embedded PNG base64)    │
   │  • best_model_<problem_type>.joblib                  │
   └─────────────────────────────────────────────────────┘

External libraries (process-local, not network services):
  scikit-learn, numpy/pandas, joblib, matplotlib/seaborn, jinja2,
  optional xgboost/lightgbm

Auth flow: (none — local CLI, user runs with OS file permissions)

WebSocket flow: (none)
```

---

## 5. Problem Statement

| Question | Answer (from codebase + README) |
|----------|----------------------------------|
| **Real-world problem** | Turning a **tabular CSV** into **compared, evaluated ML models**, optional tuning, plots, and a **persistent best model** without hand-writing pipelines for each dataset. |
| **Target user** | ML practitioners, analysts, students, and prototypers who work from **CSV files** and want a **guided CLI** (README: prototyping, Kaggle-style use, analytics, research). |
| **Manual work without this tool** | Manual EDA, choosing algorithms, writing preprocessing, training many models, picking metrics, tuning hyperparameters, plotting confusion/ROC/residuals, and writing reports — **repeated per project**. |
| **Core value** | A **single-command (or few-flag) pipeline** from CSV to **leaderboard, best model, optional HTML report, and `.joblib` artifact**, with **problem-type detection** and **large-data shortcuts** (`--sample`, `--quick`, `automl_fast`). |

---

## 6. Authentication & Authorization

| Topic | Detail |
|-------|--------|
| **Strategy** | **None.** There is no identity layer; the app reads local files and writes local outputs. |
| **Tokens** | Not applicable. |
| **Roles / permissions** | Not applicable. |
| **Auth middleware** | **Not present.** Closest “gate” is **user input validation** in `get_user_input()` / OS file access when opening `dataset`. |

---

## 7. Database Design

| Aspect | Detail |
|--------|--------|
| **Database engine** | **None** (no SQLite/Postgres/etc.). |
| **Data “model”** | In-memory **`pandas.DataFrame`** (`DatasetAnalyzer.df`) and **`pd.Series`** for `y`. |
| **Artifacts** | CSV inputs under `data/` (sample + user paths); outputs: `.html`, `.png`, `.joblib`. |
| **Relationships** | N/A (not a relational schema). Features are **columns** of one table; target is one selected column. |
| **Indexing** | Only implicit pandas indices; **no migration or seed** layer. **Sample data** is created by `generate_sample_data.py` when run as main. |

---

## 8. Error Handling & Logging

| Topic | Implementation |
|-------|----------------|
| **Global handler** | **`main()`** in `automl_cli.py` wraps the pipeline in `try/except`: `KeyboardInterrupt` → friendly message and `sys.exit(0)`; other `Exception` → `CLIFormatter.print_error(f"Error: {str(e)}")` and `sys.exit(1)`. |
| **Client error format** | **Console strings** (colorama), not JSON. |
| **Logging library** | **No** `logging` module usage in the reviewed sources; errors are **printed** via `CLIFormatter.print_error` or bare `print` in some `visualizer` except blocks. |
| **Custom error classes** | **None**; uses built-in `Exception` / `ValueError` / generic `raise Exception(...)` in `DatasetAnalyzer.load_dataset`. |
| **Per-module behavior** | `ModelTrainer._train_single_model` catches exceptions and returns failed rows so one bad model does not crash all; `HyperparameterTuner.tune_model` catches and prints tuning failures. |

---

## 9. Security Measures

| Measure | Present? | Notes |
|---------|------------|--------|
| **Input validation (CLI)** | Partial | `get_user_input` validates against `options` when provided; target column existence checked before use. |
| **Path traversal / untrusted paths** | Not hardened | `dataset` is passed to `pd.read_csv(self.filepath)` with **no sanitization** — acceptable for **trusted local use**; risky if ever exposed as a service. |
| **Rate limiting / Helmet / CORS** | N/A | No HTTP server. |
| **SQL/NoSQL injection** | N/A | No query language to DB. |
| **Secrets / env vars** | Not used | No API keys in code. |
| **Dependency safety** | Standard pip set | Optional imports reduce hard failures. |
| **Windows console encoding** | Yes | `cli_utils.py` sets **UTF-8** on `stdout`/`stderr` on win32 to reduce `UnicodeEncodeError` for emoji/box-drawing. |

---

## 10. Project Structure

Tree (major paths; depth ~3):

```
AutoML-CLI/
├── automl_cli.py           # Main CLI orchestration
├── automl_fast.py          # Wrapper: default large-data flags → main()
├── cli_utils.py            # Colors, CLIFormatter, ProgressTracker, print_banner
├── dataset_analyzer.py      # Load CSV, EDA, problem type, X/y split
├── preprocessor.py          # Imputation, label encoding, scaling
├── model_selector.py        # Portfolio of sklearn (+ optional xgb/lgb) models
├── model_trainer.py         # Train/evaluate, parallel, CV, save joblib
├── hyperparameter_tuner.py  # RandomizedSearchCV / GridSearchCV
├── visualizer.py            # Matplotlib/seaborn PNG exports
├── report_generator.py       # Jinja2 HTML reports with base64 images
├── generate_sample_data.py   # Creates sample CSVs in data/
├── test_portfolio.py         # Subprocess integration tests
├── requirements.txt          # Python dependencies (lower bounds)
├── PROJECT_DOCUMENTATION.md   # This file
├── data/                     # Sample / user CSVs (e.g. house_prices.csv)
├── reports/                  # Generated HTML reports
├── visualizations/           # Generated PNGs (when --visualize/--report)
├── README.md, LICENSE, *.md  # Docs / portfolio notes
└── .gitignore
```

| Path | Purpose |
|------|---------|
| `automl_cli.py` | End-to-end workflow and CLI definition. |
| `automl_fast.py` | Convenience defaults for large datasets. |
| `cli_utils.py` | Terminal UX (colors, tables, steps, banner). |
| `dataset_analyzer.py` | Ingestion and problem-type inference. |
| `preprocessor.py` | Feature/target preprocessing for sklearn. |
| `model_selector.py` | Which estimators enter the bake-off. |
| `model_trainer.py` | Training, metrics, best model, persistence. |
| `hyperparameter_tuner.py` | Post-hoc tuning of the winner. |
| `visualizer.py` | Static charts for evaluation and reporting. |
| `report_generator.py` | Single-file HTML deliverable. |
| `generate_sample_data.py` | Reproducible demo CSVs. |
| `test_portfolio.py` | Quick automated runs over sample data. |
| `data/` | Input CSV storage (samples + user data). |
| `reports/`, `visualizations/` | Default output directories for artifacts. |

---

## Summary

**AutoML-CLI** is a **Python 3.8+ command-line tool** built on **pandas, scikit-learn, joblib**, optional **XGBoost/LightGBM**, **matplotlib/seaborn**, and **Jinja2**, with dependencies **`rich` / `plotly` / `kaleido` declared but unused in `.py` code**. It exposes **no REST API and no authentication**; “endpoints” are **CLI flags** on `automl_cli.py` and the **`automl_fast.py` wrapper**. Data lives in **CSV files and memory**; persistence is **HTML/PNG/joblib on disk**, not a database.
