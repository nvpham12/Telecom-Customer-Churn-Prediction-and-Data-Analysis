# Telecom Customer Churn Prediction

## Goal
Identify high-risk customers for a telecom provider to enable proactive retention strategies.

## Approach
- Cleaned & transformed synthetic churn dataset; encoded categorical features, transformed skewed distributions.
- Trained XGBoost models: baseline, hyperparameter-tuned, and SMOTE-balanced for class imbalance.
- Built Tableau dashboard to visualize churn by tenure, contract type, and other features.

## Key Results
- Best recall: SMOTE model (+42% recall for churning customers vs tuned model).
- Found 90% of churned customers were on month-to-month contracts.
- Early churn risk highest in first month; internet service and electronic check payment linked to higher churn.

## Links
- [Jupyter Notebook.](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction/blob/main/Telco%20Customer%20Churn%20Prediction.ipynb)  
- [Tableau Dashboard](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17551339538610/Dashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- [Technical Report](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction/blob/main/Technical%20Report.md)

## Data Source and License
- Dataset: Telco Customer Churn
- Authors: scottdangelo
- Source: [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d)
- License: Apache License 2.0
- Reference: IBM. (2019). Telco Customer Churn for Watson Studio.
