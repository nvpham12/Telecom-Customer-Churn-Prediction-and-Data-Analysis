# Project Overview
This project demonstrates predicting customer churn using synthetic telecom data with the XGBoost algorithm. The aim is to showcase techniques for identifying customers at risk of leaving through thorough exploratory analysis, model tuning, class balancing, and interactive visualization with Tableau. All steps use reproducible workflows and accessible via colorblind friendly visual design.

For technical details, see the Jupyter Notebook.  
View the interactive dashboard [here](https://public.tableau.com/views/TelecomCustomerChurnDashboard_17515012306460/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).

# Tools & Technologies
- **Pandas** – data manipulation and cleaning
- **NumPy** - data transformation
- **Matplotlib / Seaborn** – EDA and visualizations
- **Scikit-learn** – data preprocessing and model evaluation
- **XGBoost** – tree-based classification and feature importance visualization
- **GridSearchCV** – hyperparameter tuning
- **Imbalanced-learn (imblearn)** - data balancing via SMOTE
- **Tableau** – interactive dashboarding

# Data
- The data is sourced from IBM's Base Samples. It contains synthetic information on telecom customers such as Contract Type, Monthly Charges, Tenure, and whether they churned.
- The dataset was obtained from [here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- Feature Definitions:
  - CustomerID: A unique ID that identifies each customer.
  - Senior Citizen: Indicates if the customer is 65 or older: (Yes/No)
  - Dependents: Indicates if the customer lives with any dependents: (Yes/No)
  - Tenure: Indicates the number of months that the customer has been with the company.
  - Phone Service: Indicates if the customer subscribes to home phone service with the company: (Yes/No)
  - Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: (Yes/No)
  - Internet Service: Indicates if the customer subscribes to Internet service with the company: (No/DSL/Fiber Optic/Cable)
  - Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: (Yes/No)
  - Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: (Yes/No)
  - Device Protection: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: (Yes/No)
  - Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: (Yes/No)
  - Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: (Yes/No)
  - Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: (Yes/No)
  - Contract: Indicates the customer’s current contract type: (Month-to-Month/One Year/Two Year)
  - Paperless Billing: Indicates if the customer has chosen paperless billing: (Yes/No)
  - Payment Method: Indicates how the customer pays their bill: (Bank Transfer/Credit Card/Electronic Check/Mailed Check)
  - Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
  - Total Charges: Indicates the customer’s total charges.
  - Churn: (Yes/No)

# Exploratory Data Analysis
## Data Cleaning
- Missing values are imputed. These were missing values for monthly charges for new customers (on their first month) and were imputed using their total charges. Therefore, this will not cause data leakage when done before splitting the data later.
- Duplicates and unreasonable values are checked for.
- Data types are changed to appropriate formats for manipulation and modeling.
- Extra whitespace is removed from categorical variables.

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

# Data Preprocessing
- Total Charges is removed due to high correlation with other features.
- Some redundant features are removed because they depend on other features. For example, Multiple Lines can take on the value of 'No Service,' which implies the value for the Phone Service feature.
- Yeo-Johnson Transformation is applied to numerical features to handle skewed features.
- Categorical Variables are encoded using binary encoding and dummy variable encoding. The first column is dropped when dummy variable encoding to avoid the dummy variable trap, an issue that arises where the dummy variables are highly correlated to each other.
  
# Modeling
XGBoost will be used as the model to predict customer churn. XGBoost is a gradient-boosted tree algorithm that iteratively trains weak learners to correct previous errors. It supports classification and regression tasks; here it is applied for churn classification.

The evaluation metric best suited for the situation is aucpr. This is the area under the Precision-Recall Curve and it was selected since the data is imbalanced and correctly predicting whether customers will churn is top priority. 

3 Models were developed using different techniques:

## Base XGBoost Model
- This is the Base Model that uses all independent features without tuning or balancing.

![base_model_cm](https://github.com/user-attachments/assets/9a5a4bea-41ad-4030-808a-b13b6521e31f)

The Base Model correctly predicts whether a customer will churn with the accuracy of a coin toss. Because the data is imbalanced (most customers don't actually churn), the model heavily favors predicting "won't churn."

![base_model_report](https://github.com/user-attachments/assets/f842d79c-d610-4958-91f5-a5d68b219aa5)

The Base Model has high precision, recall, and f1-score for customers who don't churn. However those metrics fall significantly when it comes to customers who do churn. For churn prediction tasks, the most important metric is recall of churned customers. The accuracy is good, but most of this accuracy comes from predicting that customers won't churn.

## Tuned Model
- This model has hyperparameter tuning applied to the features in the base model.
- Hyperparameters include learning rate, number of trees, max depth, subsample ratio, column sample by tree, minimum child weight, gamma (minimum loss reduction), alpha (L1 regularization weight), lambda (L2 regularization weight), and negative & postive weights.
- Hyperparameters were tuned using GridSearch, which tested different parameter combinations to identify the set of parameters with the best model performance.

![tuned_model_cm](https://github.com/user-attachments/assets/5d9a0afe-ecf4-4108-b74f-07b2d598b3db)

After tuning, the model more often correctly predicts whether customer will or will not churn and makes fewer incorrect predictions indicating better overall accuracy and performance than the Base Model.

![tuned_model_report](https://github.com/user-attachments/assets/021383bd-3010-47c0-a4d0-2543d03b6f25)

The Tuned Model has slightly increased metrics across the board, indicating better overall performance than the base model. However, it is still biased towards predicting that customers won't churn and recall for predictions of customers who will churn has not risen much.

![tuned_feature_importance](https://github.com/user-attachments/assets/1bbb6d04-8212-4447-909a-4a6eb08770ac)

The most important features in the model are the contract type. Feature importance was computed using gain. The next most important features are Payment Method, Device Protection, Online Security, Tenure, and Monthly Charges.

## SMOTE Model
- For this model, the training data was first balanced using Synthetic Minority Oversampling Technique (SMOTE), a method to balance data by generating some synthetic data for the minorities from existing data.
- Hyperparameter tuning was then applied to this model after data balancing to determine optimal parameters for the model.
- The negative & postive weights parameter was removed from hyperparameter tuning since this is used for imbalanced data and SMOTE was already applied to the training data.

![smote_cm](https://github.com/user-attachments/assets/5bac422a-9399-47b0-b2f7-3c7750c043bd)

The SMOTE Model has more correct predictions of when a customer will churn. It also makes fewer predictions of "Won't Churn" when a customer acutally churns. However, it makes fewer correct predictions of when a customer won't churn and makes many more mistakes where it predicts a customer will churn when they actually remain.

![smote_report](https://github.com/user-attachments/assets/87553ef4-4b05-4b10-a5ef-4b93673af28a)

The recall for churning customers increased significantly, up around 42% from the Tuned Model (from 0.54 to 0.77). However this came at the tradeoff of lower precision and recall for non-churning customers and lower precision for churning customers. Overall accuracy was also lowered by 0.05.

![smote_feature_importance](https://github.com/user-attachments/assets/78ba8269-1b74-4662-be38-4d859af6a409)

Contract Type and Online Security are significantly more important to the SMOTE Model than other features.

## Model Comparison Summary Table
| Model       | Accuracy | Churn Recall | Non-Churn Recall | Notes |
|-------------|----------|--------------|------------------|-------|
| Base        | High     | Low (0.32)   | High             | Strong bias toward "no churn" |
| Tuned       | Higher   | Slightly improved | Strong        | Overall improvement from Base Model |
| SMOTE       | Lower    | High (0.77)  | Moderate         | Best for recall, most useful for churn prediction |

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
