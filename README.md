# ML Project — Pakistan E-Commerce Fulfillment Audit

**Student:** Umar Farooq &nbsp;|&nbsp; **Roll No:** bcsf23m503  
**Course:** Machine Learning Project  
**University:** PUCIT

---

## Research Question

> *To what extent can ensemble machine learning models, augmented by Explainable AI (SHAP), predict the probability of order fulfillment failure in Pakistan's digital economy — and which factors are responsible?*

This project investigates the **PKR 858M Inventory Crisis**: a systemic pattern where high-value orders are reserved in Pakistan's largest e-commerce platform but never fulfilled, locking up inventory without generating revenue. Each phase builds toward a complete Explainable Fulfillment Audit system.

---

## Repository Structure

```
ml-project-pakistan-ecommerce/
│
├── README.md                               ← you are here
│
├── phase1/
│   └── ML_Project_Phase1_Umar_Farooq.py   ← dataset selection & problem definition
│
├── phase2/
│   └── ML_Project_Phase2_Umar_Farooq.py   ← EDA, preprocessing, feature engineering
│
├── phase3/                                 ← coming soon
│   └── ML_Project_Phase3_Umar_Farooq.py   ← ensemble model + SHAP risk attribution
│
└── data/
    ├── Pakistan_Ecommerce_Audit_Ready.csv  ← cleaned dataset (output of Phase 2)
    └── Pakistan_Ecommerce_RFM.csv          ← RFM customer table (output of Phase 2)
```

> **Note:** The raw dataset is not committed to this repo due to file size.  
> Download it from Kaggle: [Pakistan's Largest E-Commerce Dataset](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset)  
> Place it at: `data/raw/Pakistan Largest Ecommerce Dataset.csv`

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — zusmani](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset) |
| Raw rows | 1,048,575 |
| Active rows (after cleanup) | ~582,000 |
| Original columns | 26 |
| Date range | March 2016 – August 2018 |
| Platform | Pakistan's largest e-commerce marketplace |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `item_id` | int | Unique order line identifier |
| `status` | object | Order outcome — `complete`, `canceled`, `refunded`, `pending`, etc. |
| `created_at` | datetime | Timestamp when the order was placed |
| `sku` | object | Stock keeping unit — identifies the product |
| `price` | float | Unit price of the item (PKR) |
| `qty_ordered` | int | Number of units ordered |
| `grand_total` | float | Total order value (PKR) |
| `increment_id` | object | String transaction label (dropped — redundant) |
| `category_name_1` | object | Top-level product category |
| `sales_commission_code` | object | Internal commission marker (dropped — 23.5% missing) |
| `discount_amount` | float | Discount applied to the order |
| `payment_method` | object | Method used — `cod`, `easypay`, `jazzcash`, `bankalfalah`, etc. |
| `Working Date` | datetime | Working day the order was processed |
| `BI Status` | object | Simplified status label (dropped — derivable from `status`) |
| `MV` | object | String copy of `grand_total` (dropped — redundant) |
| `Year` | int | Year extracted from Working Date |
| `Month` | int | Month extracted from Working Date |
| `M-Y` | object | Month-Year label (dropped — derivable from datetime) |
| `FY` | object | Financial year label (dropped — derivable from datetime) |
| `Customer ID` | object | Anonymised customer identifier |
| `Customer Since` | datetime | Date the customer first registered |

**Engineered columns added in Phase 2:**

| Column | Type | Description |
|---|---|---|
| `order_hour` | int | Hour the order was placed (0–23) |
| `order_day` | int | Day of week (0 = Monday, 6 = Sunday) |
| `order_month` | int | Month of year |
| `order_year` | int | Year |
| `year_month` | Period | YYYY-MM period for time-series grouping |
| `is_high_value` | int (0/1) | 1 if price > dataset median |
| `payment_group` | object | `Digital` (Easypay/JazzCash/etc.) or `COD` |
| `target_failure` | int (0/1) | **Target variable** — 1 = canceled or refunded |

---

## Project Phases

### ✅ Phase 1 — Dataset Selection & Problem Definition
`phase1/ML_Project_Phase1_Umar_Farooq.py`

Selecting the dataset, defining the research question, and scoping the three audit threads:
- Fintech Audit (payment gateway failures)
- Inventory Crisis Audit (unsold high-value SKUs)
- Loyalty Risk Index (customer churn prediction)

---

### ✅ Phase 2 — EDA, Preprocessing & Feature Engineering
`phase2/ML_Project_Phase2_Umar_Farooq.py`

Full exploratory analysis on 584,524 transactions covering three years of Pakistani e-commerce activity.

**Statistical analysis performed:**
- `head()`, `tail()`, `info()`, `describe()` on numeric and categorical columns
- Missing value audit with spatial heatmap
- Duplicate row check
- dtype verification and datetime parsing

**Visualisations produced:**

| Plot | What it shows |
|---|---|
| Missing value heatmap | WHERE nulls sit across columns, not just totals |
| `grand_total` histogram (raw + log) | Heavy right skew — justifies using median, not mean |
| `qty_ordered` histogram | Most orders are single-item; bulk outliers flagged |
| Correlation matrix | Linear relationships between numeric features and the target |
| Order status bar chart | Scale of the cancellation problem — 141,434 failed orders |
| Cancellation rate by category | Rate per category (not raw count) to remove volume bias |
| Price × qty scatter (Plotly) | Interactive — price outliers coloured by category |
| Log-scale box plot by category | Extreme upper-tail outliers in Mobiles & Tablets |
| Category volume bar chart | Which categories drive the most orders |
| Dual-axis hourly chart | Separates order volume from failure rate — night-time spike |
| Hour × weekday failure heatmap | Shows Friday night is the worst hot zone |
| Payment method failure rate | Easypay fails well above the dataset average |
| Stacked area time-series | Volume + failure rate trend across 2016–2018 |
| Monthly / yearly pie charts | Seasonal order distribution |
| Repeat vs new customer bar | 56.28% of orders from repeat customers |
| RFM scatter | Customer segments in Recency × Frequency space |

**Preprocessing steps (all justified in notebook):**
- Dropped 5 fully empty unnamed columns
- Dropped `sales_commission_code` — 23.5% missing, no domain value
- Dropped rows with nulls in the 4 audit-critical columns (<0.1% of data)
- Removed 5 redundant columns: `increment_id`, `MV`, `BI Status`, `M-Y`, `FY`
- Standardised column names
- Re-parsed datetime columns with `errors='coerce'`
- Engineered 8 new features (see table above)
- Computed RFM customer profiles
- Saved two output CSVs for Phase 3

**Key findings from Phase 2:**

- **Night-time friction** — failure rate spikes 1 AM–5 AM independent of volume. Infrastructure problem, not demand. Friday night is the worst single window — post-Jummah shopping meets bank batch processing.
- **Easypay anomaly** — digital gateways fail at meaningfully higher rates than COD. The gateway is the variable, not the customer.
- **Category risk by rate** — some lower-volume categories have higher structural failure rates than headline numbers suggest.
- **Price outliers** — `grand_total` is right-skewed to 1,000,000 PKR. The extreme tail in Mobiles & Tablets is where the PKR 858M gets locked up.
- **Retention** — 56.28% repeat customers. Churn prevention has higher ROI than acquisition at this stage.

---

### 🔲 Phase 3 — Ensemble Modeling & SHAP Risk Attribution
`phase3/ML_Project_Phase3_Umar_Farooq.py` *(coming soon)*

Planned work:
- Train Random Forest and XGBoost classifiers on `target_failure`
- SHAP explainer to produce per-transaction risk scores
- Loyalty Risk Index — cluster RFM segments into Loyal / At-Risk / Churned
- Identify the top drivers of failure (e.g. payment method, order hour, price tier)

---

## Output Files

| File | Phase | Description |
|---|---|---|
| `data/Pakistan_Ecommerce_Audit_Ready.csv` | Phase 2 output | Cleaned + feature-engineered, ready for modeling |
| `data/Pakistan_Ecommerce_RFM.csv` | Phase 2 output | Per-customer RFM scores for clustering |

---

## Libraries

```
pandas    numpy    matplotlib    seaborn    plotly
scikit-learn    xgboost    shap                    ← Phase 3
```

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset)
2. Place the CSV at `data/raw/Pakistan Largest Ecommerce Dataset.csv`
3. Open the Phase 2 notebook on Kaggle or Jupyter and run all cells top to bottom
4. Outputs will be saved to the working directory as `Pakistan_Ecommerce_Audit_Ready.csv` and `Pakistan_Ecommerce_RFM.csv`

---

*Last updated: Phase 2 complete. Phase 3 in progress.*
