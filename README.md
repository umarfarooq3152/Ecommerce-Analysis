# Pakistan E-Commerce — EDA & Preprocessing (Phase 2)

**Student:** Umar Farooq &nbsp;|&nbsp; **Roll No:** bcsf23m503  
**Course:** Machine Learning Project  
**Dataset:** [Pakistan's Largest E-Commerce Dataset (2016–2018)](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset)  
**Notebook:** `ML_Project_Phase2_Umar_Farooq.py`

---

## Research Question

> *To what extent can ensemble machine learning models, augmented by Explainable AI (SHAP), predict the probability of order fulfillment failure in Pakistan's digital economy?*

This phase focuses on understanding the **PKR 858M Inventory Crisis** — a pattern of high-value orders being reserved in the system but never fulfilled, locking up inventory without generating revenue. We investigate two specific hypotheses:

- **Friction Audit** — payment gateway failures cluster at specific hours of the night due to bank settlement windows and lack of live support
- **Easypay Anomaly** — digital payment gateways fail at structurally higher rates than Cash on Delivery, independent of order price or timing

---

## Dataset Overview

| Property | Value |
|---|---|
| Raw rows | 1,048,575 |
| Active rows (after cleanup) | ~582,000 |
| Original columns | 26 |
| Columns after preprocessing | 21 + 8 engineered |
| Date range | 2016 – 2018 |
| Total canceled orders | 141,434 |

---

## What This Notebook Does

### 1. Statistical Analysis
- `head()`, `tail()`, `info()`, `describe()` on both numeric and categorical columns
- Missing value audit — heatmap showing *where* nulls sit, not just how many
- Duplicate check
- dtype verification and datetime parsing

### 2. Visual Analysis

| Plot | Purpose |
|---|---|
| Missing value heatmap | Spatial view of null patterns across columns |
| `grand_total` histogram (raw + log scale) | Confirms heavy right skew, justifies median over mean |
| `qty_ordered` histogram | Shows most orders are single-item; flags bulk outliers |
| Correlation matrix | Linear relationships between numeric features and `target_failure` |
| Order status bar chart | Overall scale of the cancellation problem |
| Cancellation rate by category | Rate (not raw count) to avoid volume bias |
| Price × qty scatter (Plotly) | Interactive view of price outliers per category |
| Log-scale box plot by category | Reveals extreme price outliers in Mobiles & Tablets |
| Category order volume bar | Which categories drive the most transactions |
| Dual-axis hourly chart | Separates order volume from failure rate by hour |
| Hour × weekday failure heatmap | Shows the night-time spike is concentrated on specific days |
| Payment method failure rate | Confirms the Easypay Anomaly with data |
| Stacked area time-series | Monthly order volume + failure rate trend 2016–2018 |
| Pie charts (month / year) | Seasonal distribution of orders |
| Repeat vs new customer bar | Customer retention split |
| RFM scatter | Customer segments visualised in Recency × Frequency space |

### 3. Preprocessing

Every step below has a written justification in the notebook explaining *why* it was done, not just *what* was done.

- Dropped 5 fully empty unnamed columns
- Dropped `sales_commission_code` (23.5% missing, zero domain relevance)
- Dropped rows with nulls in `status`, `sku`, `category_name_1`, `Customer ID` (<0.1% of data)
- Removed redundant columns: `increment_id`, `MV`, `BI Status`, `M-Y`, `FY`
- Standardised column names (stripped whitespace)
- Re-parsed datetime columns with `errors='coerce'`

### 4. Feature Engineering

| Feature | Type | Purpose |
|---|---|---|
| `order_hour` | int | Hour of day for Friction Audit |
| `order_day` | int | Day of week (0=Mon, 6=Sun) |
| `order_month` | int | Month of year |
| `order_year` | int | Year |
| `year_month` | Period | Chronological time-series grouping |
| `is_high_value` | binary | 1 if price > median — flags inventory crisis items |
| `payment_group` | categorical | Digital vs COD — Easypay Anomaly |
| `target_failure` | binary | **Target variable** — 1 = canceled/refunded, 0 = success |

An RFM table (Recency, Frequency, Monetary) is also computed and saved separately.

---

## Key Findings

**Night-time friction** — The failure rate spikes between 1 AM–5 AM independently of order volume. This is an infrastructure problem, not a demand problem. The hour × weekday heatmap shows Friday night into Saturday morning is the worst hot zone — consistent with post-Jummah shopping coinciding with bank batch processing windows.

**Easypay anomaly** — Digital gateways fail at meaningfully higher rates than COD. COD has no real-time rejection mechanism, so the comparison isolates gateway performance as the variable.

**Category risk by rate** — Cancellation *rate* (not raw count) is the right lens. Some lower-volume categories are structurally more exposed than the headline numbers suggest.

**Price outliers** — `grand_total` is heavily right-skewed (max PKR 1,000,000 vs median ~PKR 1,500). The extreme upper tail in Mobiles & Tablets is where failed high-value transactions lock up inventory.

**Retention** — 56.28% of orders come from repeat customers. The RFM scatter shows a clear VIP segment (high frequency, recent) and a churn-risk segment (high recency = long since last order).

---

## Output Files

| File | Description |
|---|---|
| `Pakistan_Ecommerce_Audit_Ready.csv` | Fully cleaned and feature-engineered dataset for Phase 3 |
| `Pakistan_Ecommerce_RFM.csv` | Per-customer RFM table for loyalty segmentation in Phase 3 |

---

## Libraries Used

```
pandas · numpy · matplotlib · seaborn · plotly
```

---

## Next Phase (Phase 3)

The engineered features (`order_hour`, `order_day`, `is_high_value`, `payment_group`, `target_failure`) and the RFM profiles feed directly into:

- **Ensemble classifier** (Random Forest / XGBoost) to predict `target_failure`
- **SHAP explainer** to attribute risk scores — identifying *why* an order is likely to fail (e.g. +40% risk from price outlier, +30% from payment method friction)
- **Loyalty Risk Index** — clustering RFM segments to identify At-Risk customers before they churn
