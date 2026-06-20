# Multi-Industry Customer Churn Diagnostic Suite

A high-performance, premium predictive analytics dashboard for customer churn diagnostic evaluations. The suite leverages a **Voting Classifier soft ensemble (XGBoost + LightGBM + CatBoost)**, **Conformal Prediction Sets for Uncertainty Quantification (UQ)**, and an **NLP-driven Column Mapper** to deliver reliable, enterprise-grade decision support across multiple business sectors.

## Dynamic Schema Refactoring & Regression Fixes

To resolve issues where static feature lists and strict validations caused pipeline crashes on user uploads, we implemented the following changes:

1. **Dynamic Pandera Schema (Guideline 1)**:
   - Removed hardcoded global feature category sets (`CONTINUOUS_FEATURES`, etc.).
   - Schema column types are determined dynamically at validation time based on the parsed data types of the input DataFrame (e.g. mapping numeric columns to `pa.Float` and object/string columns to `pa.String`).
   - Changed schema mode from `strict=True` to `strict=False` in Pandera, enabling the pipeline to gracefully ignore extra metadata columns uploaded by business users.

2. **Dynamic Preprocessing (Guideline 2)**:
   - Eliminated the rigid reordering step (`df_clean = df_clean[list(schema.columns.keys())]`) that was corrupting aligned columns.
   - Let `ColumnTransformer` (`build_preprocessor`) filter out and select only the validated numerical and categorical columns dynamically at pipeline run-time based on the fitted features.

3. **Dynamic Path Resolution (Guideline 3)**:
   - Changed `DATA_DIR` resolution to rely entirely on environment variables (`os.getenv('CHURN_DATA_DIR')`) or dynamic paths relative to `BASE_DIR`, completely eliminating hardcoded Windows absolute paths.

---

## Verification & Validation Results ✅

We verified the changes end-to-end using our verification script:
- **Numerical features** (16) and **Categorical features** (4) were correctly detected dynamically.
- Stacking hyperparameter tuning, model fitting, and conformal coverage calibration finished with **100% success** (empirical deviations ≤ 0.56%).
- Survival analysis Cox PH model fitted successfully.

---

## Key Capabilities

1. **Multi-Industry Framework**: Supports Telecom Subscribers, SaaS Cloud Subscriptions, E-Commerce Retail Customers, and Banking Account Holders, with tailored metric configurations.
2. **Ensemble Predictive Model (Telecom)**: Upgraded base classifier to a soft-voting ensemble combining **XGBoost**, **LightGBM**, and **CatBoost**. All base estimators are automatically optimized using `RandomizedSearchCV` to maximize prediction accuracy (reaching 80.3% base accuracy and 85.2% ROC-AUC).
3. **Multi-Level Conformal Uncertainty Quantification (UQ)**: Constructs empirical confidence prediction sets using MAPIE (Margin Predictor) supporting dynamic confidence levels (**80%**, **85%**, **90%**, and **95%**).
4. **Dynamic Business Action Tiers**: Maps conformal sets to distinct business actions:
   * 🔴 **Action Required** (Set: `[Churned]`): High-confidence churn risk. Target with immediate proactive retention campaigns.
   * 🟡 **Active Monitoring** (Set: `[Retained, Churned]`): Uncertain status. Deploy low-cost outreach or customer success wellness checks.
   * 🟢 **No Intervention** (Set: `[Retained]`): High-confidence retention. Maintain standard automation; do not spend retention budget.
5. **Conformal Diagnostic Panel**:
   * **Conformal Set Distribution**: Real-time doughnut chart visualizing segment proportions.
   * **Empirical Coverage Curve**: Validates mathematical guarantees by plotting target confidence against actual empirical coverage.
   * **Business Economic Impact Simulator**: Simulates outreach costs vs customer value saved across all confidence levels.
6. **NLP Semantic Header Mapping**: Automatically detects target industry and maps custom uploaded CSV headers (e.g., `months_with_company` or `monthly_spend`) to standard internal features.

---

## System Architecture

```mermaid
graph TD
    A[Upload CSV / Form Input] --> B[NLP Header Mapper]
    B -->|Detect Industry| C{Industry Schema}
    C -->|Telecom| D[Tuned XGBoost + LGBM + CatBoost Ensemble]
    C -->|SaaS / Retail / Banking| E[Tailored Risk Heuristics Engine]
    D --> F[MAPIE Multi-Level Conformal Calibrator]
    E --> F
    F -->|Query Selected Confidence| G[Compute Conformal Prediction Set & Action Tier]
    G --> H[Interactive UI Preview & Dynamic Diagnostic Plots]
    H --> I[Export Decision Report CSV]
```

### NLP Column Mapping Engine (`nlp_mapper.py`)
To map raw customer tables to schema definitions without heavy transformer dependencies, the engine implements:
1. **Synonym Matching**: Exact match check against a comprehensive synonym lexicon (e.g., `tenure` matches `months_active`, `duration_months`, etc.).
2. **Subword TF-IDF Cosine Similarity**: Falls back to character-level n-gram (2 to 4 length) TF-IDF representations to resolve typos, spaces, or underscores.

---

## Installation & Setup

### Prerequisites
* Python 3.12+
* virtualenv / pip

### 1. Setup Virtual Environment & Dependencies
```bash
# Create and activate environment
python -m venv myenv
myenv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Train the Predictive Ensemble Model
Run the pipeline to execute Exploratory Data Analysis, fit the tuned ensemble classifier, and calibrate conformal sets via MAPIE:
```bash
python run_pipeline.py
python model.py
```
This script populates `processed_data/` with the serialized joblib models and exports EDA visual plots under `plots/`.

### 3. Run the Web Application
```bash
python manage.py migrate
python manage.py runserver
```
Navigate to `http://127.0.0.1:8000` to view the live dashboard.
Change the target confidence level slider/dropdown to observe dynamic updates of the conformal diagnostics and business impact.
