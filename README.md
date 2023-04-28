# Airline-Passenger-Satisfaction-Prediction-2
INSY 695 Group 3 - Airline Passenger Satisfaction Prediction 

https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?select=train.csv

This project is a continuation of the original airline customer satisfaction project which focuses on deriving insights from the customer satisfaction dataset in order to understand the key attributes that contribute most to customer satisfaction.

#### Steps Followed:

1) Descriptive Analysis of Data.
2) Data Modelling to Predict Customer Satisfaction.
3) Define KPI (Satisfaction Ratio) and analyse previous results to maximize it.
4) Create Dashboard to summarize results
5) Do Further Causal Inference
6) Data leakage analysis
7) Imbalanced classification check
8) Model overfitting check 
9) Model tuning using TPOT and H20
10) Hyperparameter tuning using Hyperopt and Optuna
11) Unsupervisied learning - KNN
12) Model explainability using SHAP
13) Model serving using FastAPI and Docker containers
14) CICD using Github actions
15) Drift analysis
16) Dynamic data monitoring


#### Tasks Done in Descriptive Analysis of Data:
1) Cleaning column names
2) Identify categorical columns and numerical columns
3) Check missing values
4) Check Distribution of Target Variable
5) Check Distribution of Numerical Variables with respect to Target Variable
6) Check Distribution of Categorical Variables with respect to Target Variable
7) Check Relationships between categorical variables
8) Handling Categorical Variables (Ordinal Encoding on 'Class',One Hot Encoding)
9) Check Correlations and remove highly correlated feature
10) Handling missing values with SimpleImputer
11) Remove Outliers



#### Key Results:
 
 1) LigtGBM model with tuned hyperparameters using Hyperopt was chosen as the best model with an accuracy of ~97% on the test set

<img width="1125" alt="image" src="https://user-images.githubusercontent.com/47519737/235040763-4748ce6d-4873-4f09-bc91-820cf098d1e3.png">


 2) Significant drift detected in data based on the synthetic data generated using CTGAN library (see image below)
 
 <img width="1125" alt="image" src="https://user-images.githubusercontent.com/47519737/235040662-c18e65d3-8bd1-488a-9001-847b18e80dec.png">

 3) The synthetic data was generated using the process described below

<img width="1074" alt="image" src="https://user-images.githubusercontent.com/47519737/235040917-a2de011a-0f8a-4e43-bcd3-9cc07c86133a.png">

 4) For CICD, when push to the main branch, a workflow will be triggered that will pick the pickled model, rebuild and push the docker file to Heroku and Docker Hub

<img width="337" alt="image" src="https://user-images.githubusercontent.com/47519737/235041227-a97b1605-c9cf-4f25-b45c-681853706537.png">
<img width="337" alt="image" src="https://user-images.githubusercontent.com/47519737/235041278-dce8aa50-aaa8-477f-926a-52652e72e02e.png">


 

Power BI dashboard:

<img width="1203" alt="Screenshot 2023-02-26 at 2 40 03 PM" src="https://user-images.githubusercontent.com/47519737/221433187-d3e59d78-3800-4915-a516-d1391938365f.png">




#### Tasks Done in Defining KPI (Satisfaction Ratio) and analyse previous results to maximize it:
1) We want to increase customer satisfaction rate, but need to define a single metric that we want to focus on
2) New KPI is Satisfaction Ratio = Satisfaction / (Satisfaction + Neutral or NO Satisfaction )
3) We look at the relation between this KPI and important features of the most accurate model. 
4) Analyse the Relation between Flight Distance and Satisfaction Ratio
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/ba1.png)

We can see that the satisfaction ratio is lower in low distance flights, so our maximum focus should be on them.

5) Analyse Relation between Inflight_wifi_service and Satisfaction Ratio
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/ba2.png)

We see that the satisfaction ratio increase with better wifi service, therefore we should focus towards providing better wifi service.

6) Analyse the Relation between Departure Delay and Satisfaction Ratio
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/ba3.png)

We can see that the satisfaction ratio decreases as the departure delay increase, we should try to reduce the delay

#### Tasks Done in Further Causal Inference:
1) We are interested in which features of the flight would cause customer satisfactory / unsatisfactory. Hence we will not consider information about
   Customer: 'Age', 'Gender_Male', 'Customer_Type_Loyal Customer',
   Class: 'Class' and 'Type_of_Travel_Business travel'.
   'Flight_Distance', 'Departure_Delay_in_Minutes': Most of the time it is out of our control.
2) For each feature, we want to answer the question "What if we provide better service for this feature, will my satisfaction level increase?"
   We consider a rating larger than 3 as good enough service, and a rating less or equal than 3 as need to improve.
3) Convert all features into boolean
4) Check Feature Importance - We want to see if the feature with high feature importance actually has high causal effect as well
5) Implement S-Learner using XGBCLassifier
6) Implement S-Learner using LGBMCLassifier
7) Implement T-Learner using XGBCLassifier
8) Implement T-Learner using LGBMCLassifier
9) Implement X-Learner using XGBClassifier
10) Implement X-Learner using LGBMCLassifier

![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/Causal_results.png)

The learners' results are nearly identical, with Online_boarding showing the greatest ATE, trailed by Leg_room_service and Inflight_wifi_service. To boost customer satisfaction, the airline ought to concentrate on enhancing its performance in these areas of service.


#### Team Members:
Dreama Wang - 261112206

Nishi Nishi - 261078870

Riley Zhu - 261094733

ShanShan Lao - 261072808

Vibhu Bhardwaj - 261113187

Utkarsh Nagpal - 261071466

Mathilda Zhang - 261112212

Haoying Xu - 261109413

Oscar Montemayor - 261082079



![bao-menglong--FhoJYnw-cg-unsplash](https://user-images.githubusercontent.com/47519737/219964307-0b876e94-6e03-4b4d-b31f-557d57b354dd.jpg)

