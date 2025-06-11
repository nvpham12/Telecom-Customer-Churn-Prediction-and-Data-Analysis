# Project Overview
This is a project on that applies Extreme Gradient Boosting (XGBoost) to predict whether a telecom customer will cancel their plan or subscription (churn). For more technical details, refer to the Jupyter Notebook file.

# Data
- The data is sourced from IBM's Base Samples. It contains information on telecom customers and churn such as Contract Type, Monthly Charges, and Tenure.
- A copy of the dataset can be downloaded from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- More information about the data can be found here: https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

# Exploratory Data Analysis
- Missing values are imputed.
- Duplicates and unreasonable values are checked for.
- Data types are changed to appropriate formats for manipulation and modeling.
- Extra whitespace is removed from object data types.

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

# Data Preprocessing
- Total Charges is removed due to high correlation with other features.
- Some redundant features are removed because they depend on other features. For example, Multiple Lines can take on the value of 'No Service,' which implies the value for the Phone Service feature.
- Yeo-Johnson Transformation is applied to numerical features to handle skewed features.
- Categorical Variables are encoded using binary encoding and dummy variable encoding. The first column is dropped when dummy variable encoding to avoid the dummy variable trap, an issue that arises where the dummy variables are highly correlated to each other.
  
# Modeling
XGBoost will be used as the model to predict customer churn. XGBoost is a tree type model that trains first using a base learning, then computes the errors. It will then train to reduce those errors and compute new errors, repeating the process until a stopping condition is met. XGBoost can be used for both classification and regression tasks. For this project, churn prediction is a classification task. 

The evaluation metric best suited for the situation is 'aucpr,' the Area Under the PR Curve since the data is imbalanced and it's most important to correctly predict whether customers will churn. 

3 Models were developed using different techniques.

## Base XGBoost Model
- This is the Base Model that uses all independent features without tuning or balancing.

![base_model_cm](https://github.com/user-attachments/assets/9a5a4bea-41ad-4030-808a-b13b6521e31f)
The Base Model correctly predicts whether a customer will churn with the accuracy of a coin toss. Because the data is imbalanced (most customers don't actually churn), the model heavily favors predicting "won't churn."

![base_model_report](https://github.com/user-attachments/assets/f842d79c-d610-4958-91f5-a5d68b219aa5)
The Base Model has high precision, recall, and f1-score for customers who don't churn. However theose metrics fall significantly when it comes to customers who do churn. For churn prediction tasks, the most important metric is recall of churned customers. The accuracy is good, but most of this accuracy comes from predicting that customers won't churn.

## Tuned Model
- This model has hyperparameter tuning applied to the features in the base model.
- Hyperparameters were tuned using GridSearch, which tested different parameter combinations to identify the set of parameters with the best model performance.

![tuned_model_cm](https://github.com/user-attachments/assets/5d9a0afe-ecf4-4108-b74f-07b2d598b3db)
After tuning, the model more often correctly predicts whether customer will or will not churn and makes fewer incorrect predictions indicating better overall accuracy and performance than the Base Model.

![tuned_model_report](https://github.com/user-attachments/assets/021383bd-3010-47c0-a4d0-2543d03b6f25)
The Tuned Model has slightly increased metrics across the board, indicating better overall performance than the base model. However, it is still biased towards predicting that customers won't churn and recall for predictions of customers who will churn has not risen much.

![tuned_feature_importance](https://github.com/user-attachments/assets/1bbb6d04-8212-4447-909a-4a6eb08770ac)
The most important features in the model are the contract type. The next most important features are Payment Method, Device Protection, Online Security, Tenure, and Monthly Charges.

## SMOTE Model
- For this model, the training data was first balanced using Synthetic Minority Oversampling Technique (SMOTE), a method to balance data by generating some synthetic data for the minorities from existing data.
- Hyperparameter tuning was then applied to this model.
- scale_pos_weight was removed from hyperparameter tuning since this hyperparameter is used for imbalanced data and SMOTE was already applied to the training data.

![smote_cm](https://github.com/user-attachments/assets/5bac422a-9399-47b0-b2f7-3c7750c043bd)
The SMOTE Model has more correct predictions of when a customer will churn. It also makes fewer predictions of "Won't Churn" when a customer acutally churns. However, it makes fewer correct predictions of when a customer won't churn and makes many more mistakes where it predicts a customer will churn when they actually don't.

![smote_report](https://github.com/user-attachments/assets/e8eb2dfa-88a7-4876-bed0-641217bd6084)
The recall for churning customers increased significantly. However this came at the tradeoff of lower precision and recall for non-churning customers and lower precision for churning customers. Overall accuracy was also lowered.

![smote_feature_importance](https://github.com/user-attachments/assets/78ba8269-1b74-4662-be38-4d859af6a409)
Contract Type and Online Security are significantly more important to the SMOTE Model than other features.

# Churn Distribution Plots

## Contract Type
![churn_by_contract](https://github.com/user-attachments/assets/eb0f5920-ebad-4a60-9272-42b218516766)
The number of churners is negatively correlated with contract length. Significantly more people churned when they had Month to Month Contracts than when they had a 1 year or 2 year contract.

## Internet Service
![churn_by_internet_service](https://github.com/user-attachments/assets/00294fd7-ec85-4175-8fe3-0dfbc756535d)
There are higher proportions of customers who churn when they have internet service. Fiber Optic users also have a higher proportion of churners than DSL users. This could indicate problems with the Fiber Optic cable services offered by the company.

## Payment Methods 
![churn_by_payment_method](https://github.com/user-attachments/assets/29a2c1b6-c83f-4cff-b3db-14b33d5e1b4a)
Customers who pay via electronic checks have higher proportions of churning customers than those who use other payment methods (and those other payment methods have roughly the same levels of churn). This could indicate problems with the electronic check payment method.

## Tenure
![churn_by_tenure](https://github.com/user-attachments/assets/c34ea95f-1528-410d-8fe5-e1deccaf4fdf)
Customers tend to churn the most within the first couple months of tenure. The first month is the most important with customers churning more in that particular month than in any other month.

# Recommendations
- For predicting customer churn, the SMOTE model should chosen despite having lower accuracy than the Base Model and Tuned Models. Because telecom data tends to be imbalanced with customer churn being the minority, models should be optimized for the highest recall, which the SMOTE Model has (around 50% improvement from the other 2 models).
- Since customers tend to churn significantly more after their first month with the telecom company, the company should prioritize deals and promotions that lock the customer into a 1 year or 2 contract. The company can offer customers discounted or free phones or discounts through statement credits to win over new customers, while locking them into 1 or 2 year contracts
- Given how less likely customers are to churn after a year or 2 of tenure, the company is recommended to be more generous with offers to lock in customers.

# Clarifying Questions
- Why do customers tend to churn more when they have internet services with the company? Are there issues with the company's internet services?
- Why are customers who pay with electronics checks more likely to churn than customers who pay via other methods? Do customers tend to have more issues with payment using this method?
- Why do customers tend to churn most often after the first 1 or 2 months with the company? Is the onboarding experience terrible for new customers?
