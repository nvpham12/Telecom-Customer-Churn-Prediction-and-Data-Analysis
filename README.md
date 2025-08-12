# Project Overview
This project demonstrates predicting customer churn using synthetic telecom data with the XGBoost algorithm. The aim is to showcase techniques for identifying customers at risk of leaving through thorough exploratory analysis, model tuning, class balancing, and interactive visualization with Tableau. All steps use reproducible workflows and accessible via colorblind friendly visual design.

For technical details, see the Jupyter Notebook.  
View the Tableau dashboard [here](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17515012306460/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).

# Tools & Technologies
- **Pandas** – data manipulation and cleaning
- **NumPy** - data transformation
- **Matplotlib / Seaborn** – EDA and visualizations
- **Scikit-learn** – data preprocessing and model evaluation
- **XGBoost** – tree-based classification and feature importance visualization
- **GridSearchCV** – hyperparameter tuning
- **Imbalanced-learn (imblearn)** - data balancing via SMOTE
- **Tableau** – interactive dashboarding

# Data Source and License
- Dataset: Telco Customer Churn
- Authors: scottdangelo
- Source: [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d)
- License: Apache License 2.0
- Reference: IBM. (2019). Telco Customer Churn for Watson Studio.

# Data Cleaning
- There were missing values for monthly charges for new customers, which were imputed using their total charges.
- Data types are changed to appropriate formats for manipulation and modeling.
- Extra whitespace is removed from categorical variables.

# Exploratory Data Analysis
## Heatmap
![numerical_heatmap](https://github.com/user-attachments/assets/dd2fdb5e-443f-47d7-8f65-f738c01feec4)

Tenure has high 0.83 correlation with Total Charges, which is reasonable considering the longer a customer stays with the company, the larger the accumulation of their charges.
Monthly Charges has a moderate 0.65 correlation with Total Charges. This likely means that customers have a tendency to change their plans, and by extension, their Monthly Charges rather than sticking to the same plan.

## Categorical Variable Counts
![categorical_var_counts](https://github.com/user-attachments/assets/b4e15e8a-ac3d-4863-b858-d41b8f8d7aa9)

Most of the variables are imbalanced. While this is normal in the telecom industry, since most customers won't churn, this may cause some issues with model performance and results. We'll later model the data with and without balancing the data.

## Numerical Variable Distributions
![numerical_distributions](https://github.com/user-attachments/assets/d35d5498-efed-433f-b363-c9045a652444)

These variables do not have normal distributions. Tenure and Monthly Charges are multimodal, while Total Charges is right skewed. These features will require a transformation to deal with the skew before modeling.

## Tableau Dashboard
- View the dashboard [here](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17515012306460/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).
- The visualizations focus on two informative continuous variables — Monthly Charges and Tenure — which show clear churn-related trends.
- Filters for categorical features allow dynamic segmentation to explore how churn patterns shift across different customer groups.

## Dashboard Preview
<img width="1799" height="1199" alt="telco-dashboard" src="https://github.com/user-attachments/assets/56a58c43-f4f8-43d1-a15e-56a0ec7358c3" />

# Data Preprocessing
- Total Charges is removed due to high correlation with other features.
- Some redundant features are removed because they depend on other features. For example, Multiple Lines can take on the value of 'No Service,' which implies the value for the Phone Service feature.
- Yeo-Johnson Transformation is applied to numerical features to handle skewed features.
- Categorical Variables are encoded using binary encoding and dummy variable encoding. The first column is dropped when dummy variable encoding to avoid the dummy variable trap, an issue that arises where the dummy variables are highly correlated to each other.
  
# Churn Prediction Models
XGBoost will be used as the model to predict customer churn. XGBoost is a gradient-boosted tree algorithm that iteratively trains weak learners to correct previous errors. It supports classification and regression tasks; here it is applied for churn classification.

The evaluation metric best suited for the situation is aucpr. This is the Area Under the Precision-Recall Curve and it was selected since the data is imbalanced and correctly predicting whether customers will churn is top priority. 

3 Models were developed using different techniques:
- Base Model: This is a baseline model without tuning or data balancing.
- Tuned Model: This model uses hyperparameter tuning to increase model performance.
- SMOTE Model: This model first applies Synthetic Minority Oversampling Technique (SMOTE), balancing the data by generating synthetic data for the minority classes. Then, hyperparameter tuning is performed.

# Model Performance Metrics
- A classification report is generated for each model, showing model performance metrics such as Precision, Recall, Accuracy, and F1-Score.
- A confusion matrix is generated for each model to show how its predictions compare to the actual sentiment labels.
- The ROC-AUC score is computed for each model to evaluate the models' effectiveness.

## Metrics Table

| Model  | Sentiment    | Precision | Recall | F1-Score |
|--------|--------------|-----------|--------|----------|
| Base   | Won't Churn  | 0.83      | 0.88   | 0.86     |
|        | Will Churn   | 0.61      | 0.51   | 0.56     |
| Tuned  | Won't Churn  | 0.85      | 0.92   | 0.88     |
|        | Will Churn   | 0.70      | 0.54   | 0.61     |
| SMOTE  | Won't Churn  | 0.90      | 0.77   | 0.83     |
|        | Will Churn   | 0.55      | 0.77   | 0.64     |

- The Tuned Model shows modest, consistent improvements over the Base Model across all metrics.
- Compared to the Tuned Model, the SMOTE Model achieves substantially higher recall for churning customers, but at the cost of lower precision.
- The Tuned Model outperforms SMOTE in precision for churning customers and recall for non-churning customers, indicating a more balanced trade-off.

---
## Accuracy, ROC-AUC, and Macro-Average Metrics Table

| Model  | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
|--------|-----------|--------|----------|----------|---------|
| Base   | 0.72      | 0.70   | 0.71     | 0.78     | 0.83    |
| Tuned  | 0.77      | 0.73   | 0.75     | 0.82     | 0.86    |
| SMOTE  | 0.72      | 0.77   | 0.64     | 0.77     | 0.85    |
> Note: This table uses macro-averages for precision, recall, and f1-score.
- The Base Model is consistently outperformed by the Tuned Model.
- The Tuned Model has the strongest overall balance.
- The SMOTE Model boosts recall with tradeoffs in precision and accuracy.

# Churn Proportion Analysis
Proportions of churned and retained customers are visualized across key features:

## Contract Type
![churn_by_contract](https://github.com/user-attachments/assets/7731f475-183a-4e91-9e0e-21b3be36ebe2)

Customers with month-to-month contracts have the highest churn rates, while those on 1-year and 2-year contracts are significantly less likely to churn. This suggests that contract length is negatively associated with churn, and long-term contracts may help retain customers.

![churn_contract_pie](https://github.com/user-attachments/assets/4ec9bc1a-c1aa-4f57-896b-20399ec3d5c4)

Of all churning customers, around 90% are on month-to-month contracts. This drops to around 9% for customers on 1 year contracts. For the company, it is of the utmost importance to get customers on 1 or 2 year contracts, which will drastically reduce churn rate.

## Internet Service
![churn_by_internet_service](https://github.com/user-attachments/assets/72ac1b6a-1aa0-42df-8da1-ecc28158f810)

Churn is also higher among customers who have internet service with the company. Churn rate is even higher for those who have fiber optic connections compared to DSL. This may reflect potential service quality or satisfaction issues with fiber optic offerings and even the internet service in general.

## Payment Methods 
![churn_by_payment_method](https://github.com/user-attachments/assets/ece176ff-cc52-4a1e-b5f2-52db3eb62343)

Customers who pay via electronic checks have higher proportions of churning customers than those who use other payment methods (and those other payment methods have roughly the same levels of churn). This could indicate high frequencies of technical issues for customers pay using electronic checks, which leads to frustration and churn.

## Tenure
![churn_by_tenure](https://github.com/user-attachments/assets/c34ea95f-1528-410d-8fe5-e1deccaf4fdf)

Customers tend to churn the most within the first couple months of tenure. The first month is the most important with customers churning more in that particular month than in any other month.

# Recommendations
- For predicting customer churn, the SMOTE model would be preferred despite having lower accuracy than the Base Model and Tuned Models. Since telecom data tends to be imbalanced with customer churn being the minority, models should be optimized for the highest recall, which the SMOTE Model has (up to 42% improvement from the Tuned Model).
- Because customers tend to churn significantly more after their first month with the telecom company, the company should prioritize deals and promotions that lock the customer into a 1 year or 2 contract.
- The company can offer customers discounted or free phones or discounts through statement credits to win over new customers, while locking them into 1 or 2 year contracts.
- Since churn rates drop substantially after the first year, the company may benefit from offering upfront incentives to encourage long-term commitments and reduce early-stage churn (which is where churn rate is the highest).

# Insights
- Customers with month-to-month contracts are ~9× more likely to churn than those on annual contracts.
- Internet service and payments via electronic checks show elevated churn risk.
- Most churn occurs within the first 4 months of a customer's tenure, with the first month having the highest churn risk.

# Clarifying Questions
- Why are customers with internet service more likely to churn?
- Are there pain points in the onboarding process?
- Do electronic check users face billing friction or service issues?

# Next Steps
- Investigate whether onboarding experience is correlated with early churn.
- Explore alternative classification algorithms such as Random Forest or Logistic Regression.
- Test different resampling strategies beyond SMOTE (e.g., ADASYN or Tomek Links).
- Add additional features such as quarters, geographic information, and download rates.
