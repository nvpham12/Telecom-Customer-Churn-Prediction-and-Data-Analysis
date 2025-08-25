# Project Background
This project demonstrates predicting customer churn using synthetic telecom data with the XGBoost algorithm. The aim is to showcase techniques for identifying customers at risk of leaving through thorough exploratory analysis, model tuning, class balancing, and interactive visualization with Tableau. All steps use reproducible workflows and accessible via colorblind friendly visual design.

## Tools & Technologies
- **Pandas** – data manipulation and cleaning
- **NumPy** - data transformation
- **Matplotlib / Seaborn** – EDA and visualizations
- **Scikit-learn** – data preprocessing and model evaluation
- **XGBoost** – tree-based classification and feature importance visualization
- **GridSearchCV** – hyperparameter tuning
- **Imbalanced-learn (imblearn)** - data balancing via SMOTE
- **Tableau** – interactive dashboarding

## Approach
- Cleaned & transformed synthetic churn dataset; handled missing values, changed data types, encoded categorical features, and transformed skewed distributions.
- Trained XGBoost models: baseline, hyperparameter-tuned, and SMOTE-balanced for class imbalance.
- Built a Tableau dashboard to visualize churn by tenure, contract type, and other features.

# Links
- [Jupyter Notebook with code](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction/blob/main/Telco%20Customer%20Churn%20Prediction.ipynb)  
- [Tableau Dashboard](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17551339538610/Dashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- [Full Technical Report](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction/blob/main/Technical%20Report.md)

# Tableau Dashboard Preview
<img width="1720" height="1190" alt="Telco Dashboard Preview" src="https://github.com/user-attachments/assets/e42571f2-7aa9-467e-93fa-f03485af9800" />

- View the dashboard [here](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17551339538610/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- The visualizations focus on Monthly Charges, Tenure, Total Revenue, and Number of Customers which show churn patterns or segments.
- Filters for categorical features allow dynamic segmentation to explore how churn patterns shift across different customer groups.

# Executive Summary
## Insights
- Churning customers with month-to-month contracts have been ~9× more likely to churn than those on annual contracts.
- Internet service and payments via electronic checks show elevated churn risk.
- Most churn occurs within the first 4 months of a customer's tenure, with the first month having the highest churn risk.

## Recommendations
- For predicting customer churn, the SMOTE model would be preferred despite having lower accuracy than the Base Model and Tuned Models. Since telecom data tends to be imbalanced with customer churn being the minority, models should be optimized for the highest recall, which the SMOTE Model has (up to 42% improvement from the Tuned Model).
- Because customers tend to churn significantly more after their first month with the telecom company, the company should prioritize deals and promotions that lock the customer into a 1 year or 2 contract.
- The company can offer customers discounted or free phones or discounts through statement credits to win over new customers, while locking them into 1 or 2 year contracts.
- Since churn rates drop substantially after the first year, the company may benefit from offering upfront incentives to encourage long-term commitments and reduce early-stage churn (which is where churn rate is the highest).

## Clarifying Questions
- Why are customers with internet service more likely to churn?
- Are there pain points in the onboarding process?
- Do electronic check users face billing friction or service issues?

## Next Steps
- Investigate whether onboarding experience is correlated with early churn.
- Explore alternative classification algorithms such as Random Forest or Logistic Regression.
- Test different resampling strategies beyond SMOTE.

# Data Source and License
- Dataset: Telco Customer Churn
- Authors: scottdangelo
- Source: [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d)
- License: Apache License 2.0
- Reference: IBM. (2019). Telco Customer Churn for Watson Studio.
