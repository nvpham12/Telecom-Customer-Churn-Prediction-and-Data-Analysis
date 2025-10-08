# Project Background
This project demonstrates data analytics and customer churn prediction using synthetic telecom data with the XGBoost algorithm. Data visualization was completed using Seaborn and Matplotlib. An interactive dashboard was created using Tableau.

## Links
### Data Analytics
- [Data Analytics Jupyter Notebook](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction-and-Analysis/blob/main/Data%20Analysis%20Telecom%20Customer%20Churn.ipynb)
- [Full Data Analytics Report](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction-and-Analysis/blob/main/Analytics%20Report.md)
### Churn Prediction
- [Data Cleaning and Churn Prediction Jupyter Notebook](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction-and-Analysis/blob/main/Churn%20Prediction%20Telecom%20Customers.ipynb)
- [Churn Prediction Model Technical Report](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction/blob/main/Technical%20Report.md)
### Tableau Dashboard
- [Tableau Dashboard](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17551339538610/Dashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Tools & Technologies
- **Pandas** – data manipulation, cleaning, and exporting cleaned data
- **NumPy** - data transformation
- **Matplotlib / Seaborn** – EDA and visualizations
- **Scikit-learn** – data preprocessing and model evaluation
- **XGBoost** – tree-based classification and feature importance visualization
- **GridSearchCV** – hyperparameter tuning
- **Imbalanced-learn (imblearn)** - data balancing via SMOTE
- **Tableau** – interactive dashboarding

## Approach
- Cleaned and transformed synthetic churn dataset: handled missing values, changed data types, encoded categorical features, and transformed skewed distributions.
- Trained XGBoost models: baseline, hyperparameter-tuned, and SMOTE-balanced for class imbalance.
- Used Matplotlib and Seaborn to generate visualizations that show churn patterns and customer distributions.
- Interpreted the visualizations, documenting and delivering actionable insights and recommendations.
- Built a Tableau dashboard to visualize churn by tenure, contract type, and other features.

## Data
- The data contains information on telecom customer churn and contains 7043 rows and 21 columns.
- Columns include customer demographics and add-on services also offered by the company.
- The data is synthetic, made and shared by IBM.

---

# Executive Summary
## Insights
- Customers with Monthly Charges ranging from \$75-\$100 are the most numerous and have the highest churn rates.
- The monthly charge brackets may represent different customer segments for targeted retention strategies.
- Most of the company's customers are new customers, making onboarding experiences very significant to reducing churn.
- Those on Month-to-Month contracts have around 14 times the churn rate of those on Two-Year contracts and around 4 times the churn rate of those on One-Year contracts.
- Around 89% of churning customers are on Month-to-Month contracts.
- Customers using the electronic payment method have higher churn rates than other payment method, which may indicate serious technical issues, poor user interface for inputting information, or lack of security for the payment method.
- Customers using the Online Security service or Tech Support have lower churn rates than those who do not use them.

## Recommendations
- Since customers tend to churn significantly more after their first month, the company should prioritize deals and promotions that lock the customer into a 1-Year or 2-Year contract. Developing a strong onboarding experience and engaging new customers is very important.
- The company can offer customers discounted or free phones or discounts through statement credits to win over new customers, while locking them into longer contracts, which have significantly lower churn rates.
- Promote add-ons and other services, especially the online security add-on. Investigate potential technical issues with the electronic payment system and encourage customers to use tech support when necessary.
- Develop targeted strategies for different customer segments, such as those with different levels of monthly spending.
- Customers without dependents have higher churn rates than those that do. Consider reducing single line prices or providing other offers to those with single lines to incentivize adding more lines.

---

# Data Source and License
- Dataset: Telco Customer Churn
- Authors: scottdangelo
- Source: [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d)
- License: Apache License 2.0
