# Algorithmic Customer Churn & Lifetime Value (LTV) Optimization Engine

An end-to-end Data Science and Machine Learning framework built in Python to predict retail banking customer churn, estimate 3-year Customer Lifetime Value (LTV), optimize marketing campaign spend using an **Expected Value (EV)** decision model, and apply **CATE Uplift Modeling** to target incremental retention with precision.

---

## 📌 Executive Summary

Traditional customer retention strategies in banking often rely on blanket marketing or naive model thresholds (e.g., targeting everyone with a predicted churn probability $> 0.50$). This leads to massive budget inefficiency:
1. **Wasted Capital on "Sure Things":** Offering incentives to customers who intended to stay anyway.
2. **Wasted Capital on "Lost Causes":** Offering incentives to customers who leave regardless of $50 offers.
3. **Negative ROI on Low-Value Accounts:** Spending $50 to retain customers with $10 balances.

This project solves this multi-layered problem by combining **Inferential Hypothesis Testing**, **Machine Learning**, **Financial Decision Modeling**, and **CATE Uplift Modeling**.

---

## 📊 Key Results & Financial Impact

| Metric / Analysis Stage | Outcome / Result | Strategic Business Impact |
| :--- | :--- | :--- |
| **Analyzed Portfolio** | **10,000 Accounts** | Processed, cleaned, and encoded retail bank customer dataset. |
| **Inferential A/B T-Test** | **$T = 29.77, p = 1.24 	imes 10^{-186}$** | Statistically proved churned clients are significantly older ($pprox 44.8$ yrs vs $37.4$ yrs). |
| **Predictive Model ROC-AUC** | **0.8647** | Balanced Random Forest achieved strong discriminative ability with **70% recall** on churners. |
| **EV Optimized Revenue** | **$7,114,572.93** | Delivered **+$3,296,171.29 MORE net profit** than traditional threshold targeting. |
| **Uplift Target Persuadables** | **114 Accounts** | Reduced offer spend from **$319,100 down to $5,700** while capturing maximum incremental ROI. |

---

## 🛠️ Project Architecture & Pipeline

```
Phase 1: EDA & Inferential Statistics
  └── Clean missing values, encode features, perform two-sample T-test on churn drivers.

Phase 2: Predictive Modeling & LTV Proxy
  └── Train balanced RandomForestClassifier, extract churn probabilities P(Churn), compute 3-Year LTV.

Phase 3: Financial Expected Value Framework
  └── Calculate Net EV = (P(Churn) * LTV * 25% Conversion) - $50 Offer Cost.

Phase 4: Uplift CATE Modeling (T-Learner)
  └── Train Control vs. Treatment estimators, isolate CATE uplift score, filter strictly for "Persuadables".
```

---

## 💡 The 4 Uplift Quadrants Identified

Out of 10,000 portfolio accounts, CATE Uplift Modeling categorized the customer base into four actionable segments:

1. **Sure Things (4,784 accounts):** Customers who stay regardless of getting an offer. *Withheld $50 offer $ightarrow$ Saved $239,200.*
2. **Do Not Disturb / Low Impact (4,210 accounts):** Low baseline risk or negligible balance return.
3. **Lost Causes (892 accounts):** High churn probability unaffected by $50 offer. *Withheld offer $ightarrow$ Saved $44,600.*
4. **Persuadables (114 accounts):** **High-value priority segment!** Incentive directly flips their retention decision.

---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.13+
* **Data Processing:** `pandas`, `numpy`
* **Statistical Analysis:** `scipy.stats`
* **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `train_test_split`, `roc_auc_score`)
* **Environment:** Jupyter Notebook / Anaconda base environment

---

## 🚀 How to Run locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Fintech-Customer-Retention-Uplift.git
cd Fintech-Customer-Retention-Uplift
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook "Algorithmic Customer Churn & Lifetime Value (LTV) Optimization Engine.ipynb"
```

---

## 📁 Data Outputs Generated

* `Churn_Predictions_Output.csv`: Dataset with predicted $P(	ext{Churn})$ scores and LTV estimates.
* `Targeted_Marketing_Campaign_List.csv`: Actionable list filtered by positive Expected Value ($	ext{EV} > \$0$).
* `Uplift_Optimized_Campaign.csv`: Complete portfolio segmented into the 4 Uplift Quadrants.
* `High_Priority_Persuadables_List.csv`: Actionable list of the top **114 Persuadables** exported for immediate marketing execution.

---

## 👤 Author
**Obaloluwa Temidayo Koyi-Kayode**
* Data Analyst | Mathematics Graduate
* Specialized in Quantitative Risk Modeling, FinTech Analytics, and Decision Optimization Systems.
