# Supply Chain Delay & Profitability Analysis

End-to-end data science project on the **DataCo Supply Chain** dataset: 180 k order-item records across 5 global regions (2015–2018). The project spans data cleaning, exploratory analysis, delay root-cause investigation, and a machine-learning classifier for late delivery prediction.

---

## Key Findings

- **54.7 % of orders arrive late** — only 45.3 % on-time delivery across all regions and shipping modes
- **Standard Class drives the most volume** (62 % of orders) and is the primary source of delays
- **Delayed orders cost ~$2 M in losses**; positive-profit orders generated $7.5 M — closing the delay gap is the single highest-leverage operational improvement
- **Random Forest achieves 74 % accuracy** (F1 0.77 on late-delivery class) using only information available at order placement, making real-time risk scoring feasible

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Data Loading & Cleaning](supply_chain/01_data_loading_and_cleaning.ipynb) | Raw CSV → clean Parquet; drops PII, zero-variance, and near-duplicate columns |
| 02 | [EDA & Business KPIs](supply_chain/02_eda_and_kpis.ipynb) | Profitability distribution, delay distribution, profit analytics by delay day |
| 03 | [Delay Analysis](supply_chain/03_delay_analysis.ipynb) | Delay rate by category/region, top drivers per region, temporal trends |
| 04 | [Predictive Modeling](supply_chain/04_predictive_modeling.ipynb) | Logistic Regression, Random Forest, and XGBoost; ROC-AUC curves; confusion matrices |

---

## Dataset

**DataCo Supply Chain Dataset** — publicly available on Kaggle.

- 180,519 raw records → 172,765 after removing cancelled shipments
- 53 raw columns → 21 after dropping PII, ID surrogates, and zero-variance fields
- Covers 5 markets: LATAM, Europe, Pacific Asia, USCA, Africa
- Top departments: Fan Shop, Apparel, Golf, Footwear

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd SupplyChainAnalysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Place DataCoSupplyChainDataset.csv inside supply_chain/

# 4. Run notebooks in order
jupyter lab supply_chain/
```

Run notebooks **01 → 02 → 03 → 04** in order. Notebook 01 produces `data/clean.parquet` which all subsequent notebooks consume.

---

## Tech Stack

| Area | Libraries |
|------|-----------|
| Data wrangling | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine learning | scikit-learn, xgboost |
| Class imbalance | imbalanced-learn (SMOTE) |
| Storage | pyarrow (Parquet) |
