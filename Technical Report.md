# Project Background
This project demonstrates customer churn prediction using synthetic telecom data with the XGBoost algorithm and interactive visualization with Tableau.

## Links
- [Jupyter Notebook](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction-and-Analysis/blob/main/Churn%20Prediction%20Telecom%20Customers.ipynb)
- [Tableau dashboard](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17551339538610/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

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

## Data
- The data contains information on telecom customer churn and contains 7043 rows and 21 columns.
- The data is synthetic and was made and shared by IBM.
- The initial schema is as follows:

| Field Name         | Type     |
|--------------------|----------|
| CustomerID         | object   |
| Gender             | object   |
| SeniorCitizen      | int64    |
| Partner            | object   |
| Dependents         | object   |
| Tenure             | int64    |
| PhoneService       | object   |
| MultipleLines      | object   |
| InternetService    | object   |
| OnlineSecurity     | object   |
| OnlineBackup       | object   |
| DeviceProtection   | object   |
| TechSupport        | object   |
| StreamingTV        | object   |
| StreamingMovies    | object   |
| Contract           | object   |
| PaperlessBilling   | object   |
| PaymentMethod      | object   |
| MonthlyCharges     | float64  |
| TotalCharges       | object   |
| Churn              | object   |

# Data Cleaning
- There were missing values for monthly charges for new customers, which were imputed using their total charges.
- Data types are changed to appropriate data types for manipulation and modeling.
- Extra whitespace is removed from categorical variables.

# Exploratory Data Analysis
## Heatmap
<img width="1000" height="600" alt="numerical_heatmap" src="https://github.com/user-attachments/assets/fe374919-abad-418b-8f4a-3cf9599852d7" />

- Tenure has high 0.83 correlation with Total Charges, which is reasonable considering the longer a customer stays with the company, the larger the accumulation of their charges.
- Monthly Charges has a moderate 0.65 correlation with Total Charges. This likely means that customers have a tendency to change their plans, and by extension, their Monthly Charges rather than sticking to the same plan.

---
## Categorical Variable Counts
<img width="1400" height="1400" alt="categorical_var_counts" src="https://github.com/user-attachments/assets/ec4a5564-f347-46c5-b176-7932d025f2e4" />

- Most of the variables are imbalanced. While this is normal in the telecom industry, since most customers won't churn, this may cause some issues with model performance and results.
- The data will be modeled both before and after balancing.

---
## Numerical Variable Distributions
<img width="1500" height="500" alt="numerical_distributions" src="https://github.com/user-attachments/assets/1e5e5f55-d3ef-46d8-903b-215e3f74e871" />

- These variables do not have normal distributions.
- Tenure and Monthly Charges are multimodal, while Total Charges is right skewed. These features will require a transformation to deal with the skew before modeling.

---
## Tableau Dashboard
<img width="1720" height="1190" alt="Telco Dashboard Preview" src="https://github.com/user-attachments/assets/e42571f2-7aa9-467e-93fa-f03485af9800" />

- View the dashboard [here](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17551339538610/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- The visualizations focus on Monthly Charges, Tenure, Total Revenue, and Number of Customers which show churn patterns or segments.
- Filters for categorical features allow dynamic segmentation to explore how churn patterns shift across different customer groups.

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
| Base   | Won't Churn  | 0.83      | 0.88   | 0.85     |
|        | Will Churn   | 0.59      | 0.49   | 0.54     |
| Tuned  | Won't Churn  | 0.85      | 0.91   | 0.88     |
|        | Will Churn   | 0.69      | 0.54   | 0.61     |
| SMOTE  | Won't Churn  | 0.90      | 0.77   | 0.83     |
|        | Will Churn   | 0.55      | 0.77   | 0.64     |

- The Tuned Model shows modest, consistent improvements over the Base Model across all metrics.
- Compared to the Tuned Model, the SMOTE Model achieves substantially higher recall for churning customers
- The Tuned Model outperforms SMOTE in precision for churning customers and recall for non-churning customers.

---
## Accuracy, ROC-AUC, and Macro-Average Metrics Table

| Model  | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
|--------|-----------|--------|----------|----------|---------|
| Base   | 0.71      | 0.69   | 0.70     | 0.78     | 0.83    |
| Tuned  | 0.77      | 0.73   | 0.74     | 0.81     | 0.86    |
| SMOTE  | 0.73      | 0.77   | 0.74     | 0.78     | 0.85    |
> Note: This table uses macro-averages for precision, recall, and f1-score.
- The Base Model is consistently outperformed by the Tuned Model.
- The Tuned Model and SMOTE Models are just as balanced.
- The Tuned Model boosts precision and accuracy with tradeoffs in recall.
- The SMOTE Model boosts recall with tradeoffs in precision and accuracy.

## Feature Importance
Feature importance is used to rank how inputs affect a model's predictions. In linear models, the most important features often have the largest weights. In tree-based or complex models, such as with the XGBoost model being used in this project, importance can reflect split frequency, information gain, or influence on prediction variance.

---
<img width="2480" height="1361" alt="tuned_feature_importance" src="https://github.com/user-attachments/assets/468a51fb-22c2-4b2d-8ca9-19da8a5b9de8" />

- The most important feature in the model is the Two-Year Contract. 
- The next most important features (in order) are Payment Method, One-Year Contract, Device Protection, Online Security, Tenure, and Monthly Charges.
---

<img width="2480" height="1361" alt="smote_feature_importance" src="https://github.com/user-attachments/assets/91f8a6eb-acc0-464d-9b35-48c94425a0e6" />

- Contract Type and Online Security are significantly more important to the SMOTE Model than other features.
- Contract Type becomes more important with both features having the highest ranks now.
- Online Security becomes more important than Device Protection, while Payment Method is far less important after data balancing.

# Conclusion
For predicting customer churn, the SMOTE model should chosen despite having lower accuracy than the Base Model and Tuned Models. Because telecom data tends to be imbalanced with customer churn being the minority, models should be optimized for the highest recall, which the SMOTE Model has (up to 42% improvement from the Tuned Model).

# Data Source and License
- Dataset: Telco Customer Churn
- Authors: scottdangelo
- Source: [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d)
- License: Apache License 2.0
- Reference: IBM. (2019). Telco Customer Churn for Watson Studio.
