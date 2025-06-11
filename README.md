# Project Overview
This is a project on that applies Extreme Gradient Boosting (XGBoost) to predict whether a telecom customer will cancel their plan or subscription (churn). For more technical details, refer to the Jupyter Notebook file.

# Data
The data is sourced from IBM's Base Samples. It contains information on telecom customers and churn such as Contract Type, Monthly Charges, and Tenure.
<br>
A copy of the dataset can be downloaded from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
<br>
More information about the data can be found here: https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

# Exploratory Data Analysis
- Missing values are imputed
- Duplicates and unreasonable values are checked for
- Data types are changed to appropriate formats for manipulation and modeling
- Extra whitespace is removed from object data types

## Heatmap
![numerical_heatmap](https://github.com/user-attachments/assets/dd2fdb5e-443f-47d7-8f65-f738c01feec4)
Tenure has high 0.83 correlation with Total Charges, which is reasonable considering the longer a customer stays with the company, the larger the accumulation of their charges.
Monthly Charges has a moderate 0.65 correlation with Total Charges. This likely means that customers have a tendency to change their plans, and by extension, their Monthly Charges rather than sticking to the same plan.

## Categorical Variable Counts
![categorical_var_counts](https://github.com/user-attachments/assets/b4e15e8a-ac3d-4863-b858-d41b8f8d7aa9)
Most of the variables are imbalanced. While this is normal in the telecom industry, since most customers won't churn, this could cause some issues with the model. We'll later model the data before and after balancing.

## Numerical Variable Distributions
![numerical_distributions](https://github.com/user-attachments/assets/d35d5498-efed-433f-b363-c9045a652444)
These variables do not have normal distributions. Tenure and Monthly Charges are multimodal, while Total Charges is right skewed.

# Preprocessing
- Yeo-Johnson Transformation is applied to numerical features since they differ in magnitude. For example, tenure ranges from 0 to 72, while Total Charges be in 100s or 1000s of dollars.
- Categorical Variables are encoded using binary encoding and dummy variable encoding. The first column is dropped when dummy variable encoding to avoid the dummy variable trap, an issue that arises where the dummy variables are highly correlated to each other.

# Modeling
XGBoost will be used as the model to predict customer churn. XGBoost is a tree type model that trains first using a base learning, then computes the errors. It will then train to reduce those errors and compute new errors, repeating the process until a stopping condition is met. XGBoost can be used for both classification and regression tasks. In this case, we have a classification task with imbalanced data and are interested in predicting the minority class. The evaluation metric best suited for the situation is 'aucpr,' the Area Under the PR Curve.

4 Models were developed using different features or techniques.

## Model 1
- This is a baseline model that ran on all features independent features. Some features such as Multiple Lines could imply the values for Phone Service.

## Model 2
- This model has applied hyperparameter tuning to the baseline model

## Model 3
- This model kept only some important features as measured by feature importance in Model 2.

## Model 4
- This model balanced the data using Synthetic Minority Oversampling Technique (SMOTE), a method to balance data by generating some synthetic data for the minorities from existing data.
- 
# Deep Dive into important features
