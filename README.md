# Airline-Passenger-Satisfaction-Prediction
INSY 695 Group 3 - Airline Passenger Satisfaction Prediction 

https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?select=train.csv

This project focusses on deriving insights from the customer satisfaction dataset inorder to understand the key attributes that contribute most to customer satisfaction.

#### Motivation for this project:

1. Airlines leave upto 1.4B USD in revenue on the table by failing to increase their customer satisfaction (source: Forrester)
2. It takes almost 200K USD per year for a company of size 1000 and $100 in revenue to run a NPS program (source: CustomerGauge)

#### Customer satisfaction is of utmost importance for airlines for several reasons:

1) Customer Loyalty: Airlines want to build a loyal customer base who choose to fly with them repeatedly. Customers who are satisfied with their experience are more likely to choose the same airline in the future, leading to increased revenue for the airline.

2) Reputation: Airlines rely heavily on their reputation, which is built on the experiences of their customers. Satisfied customers are more likely to spread positive word-of-mouth, leading to more customers and a better reputation. On the other hand, dissatisfied customers can harm the airline's reputation, leading to a decrease in business.

3) Competitive Advantage: Airlines operate in a highly competitive industry, and customer satisfaction is one of the key factors that sets them apart from their competitors. Airlines that prioritize customer satisfaction can gain a competitive advantage, attracting more customers and generating more revenue.

4) Revenue: Satisfied customers are more likely to spend more money on ancillary products and services offered by the airline, such as seat upgrades, in-flight meals, and baggage fees. In addition, they are more likely to return for future flights, leading to increased revenue for the airline.

5) Compliance: Airlines are subject to a variety of regulations and standards, including safety regulations and customer service standards. Meeting these standards is important for compliance, and satisfied customers are more likely to report positive feedback to regulatory authorities.

#### This project has 3 primary objectives:

1) Predict Customer Satisfaction and create a dashboard to visualize the results
2) Identify features that contribute most to customer satisfaction
3) Use semi-supervised learning to create labels for data

#### Expected Outcome:

1) Increased Revenue (upto USD 1.4B)
2) Cost savings in running NPS programs
3) Improved brand image and perception

#### Steps Followed:

1) Descriptive Analysis of Data.
2) Data Modelling to Predict Customer Satisfaction.
3) Define KPI (Satisfaction Ratio) and analyse previous results to maximize it.
4) Create Dashboard to summarize results
5) Do Further Causal Inference.


#### Tasks Done in Descriptive Analysis of Data:
1) Cleaning column names
2) Identify categorical columns and numerical columns
3) Check missing values
4) Check Distribtion of Target Variable
5) Check Distribtion of Numerical Variables with respect to Target Variable
6) Check Distribtion of Categorical Variables with respect to Target Variable
7) Check Relationships between categorical variables
8) Handling Categorical Vairables (Ordinal Encoding on 'Class',One Hot Encoding)
9) Check Correlations and removing highly correlated feature
10) Handling missing values with SimpleImputer
11) Remove Outliers


#### Tasks Done in Data Modelling to Predict Customer Satisfaction:
1) Standardization of numeric variables
2) Handling Missing values (The satisfaction level of each feature should range from 1 to 5.
    If the rating is 0, that means the customer did not rate for this feature.)
3) Splitting the Dataset (Train Test Split,Test Validation Split)
4) Functions for Evaluation Metrics (ROC Curve,PR Curve,Classification Score)
5) Verify Class Balance
6) Feature Selection (LASSO,RFE)
7) Based on the above results, we can drop 'Gender_Male','Gate_location','Age', 'Food_and_drink'.
8) Implement Modelling:

  Baseline Model - DummyClassifier
  
  Logistics Regression
  
  KNN
  
  Gaussian (Naive Bayees)
  
  Decision Tree
  
  Random Forest
  
  LightGBM
  
  XGBoost
  
  AdsBoost
  
9) Semi-supervised Learning:
  Label Propagation KNN
  Label Spreading

10) Evaluate model on test set
Given the results, LightGBM has the highest F1-Score. Now we test the model performance on the unseen test dataset.
  
  Results:

![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/Model_results.png)

Power BI dashboard:

<img width="1203" alt="Screenshot 2023-02-26 at 2 40 03 PM" src="https://user-images.githubusercontent.com/47519737/221433187-d3e59d78-3800-4915-a516-d1391938365f.png">


Results:

![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/git1.png)
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/git2.png)
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/git3.png)
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/git4.png)
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/git5.png)
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/git6.png)


#### Tasks Done in Defining KPI (Satisfaction Ratio) and analyse previous results to maximize it:
1) We want to increase satisfaction , but need to define a single metric that we want to focus on
2) New KPI is Satisfaction Ratio = Satisfaction / (Satisfaction + Neutral or NO Satisfaction )
3) We look at the relation between this KPI and important features of the most accurate model. 
4) Analyse Relation between Flight Distance and Satisfaction Ratio
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/ba1.png)

We can see that satisfaction ratio is lower in low distance flights so our maximum focus should be on them

5) Analyse Relation between Inflight_wifi_service and Satisfaction Ratio
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/ba2.png)

We see that satisfaction ratio increase with better wifi service , therefore we should focus towards providing better wifi service

6) Analyse Relation between Departure Delay and Satisfaction Ratio
![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/ba3.png)

We can see that satifaction ratio is decreasing as departure delay increase, we should try to decrease the delay

#### Tasks Done in Further Causal Inference:
1) We are interested in which features about the flight would cause customer satisfactory / unsatisfactory. Hence we will not consider information about
   Customer: 'Age', 'Gender_Male', 'Customer_Type_Loyal Customer',
   Class: 'Class' and 'Type_of_Travel_Business travel'.
   'Flight_Distance', 'Departure_Delay_in_Minutes': Most of the time it is out of our control.
2) For each feature, we want to answer the question "What if we provide better service for this feature, will my satisfication level increase?"
   We consider rating larger than 3 as good enough service, and rating less or equal than 3 as need to improve.
3) Convert all features into boolean
4) Check Feature Importance - We want to see if the feature with high feature importance actually has high causal effect as well
5) Implement S-Learner using XGBCLassifier
6) Implement S-Learner using LGBMCLassifier
7) Implement T-Learner using XGBCLassifier
8) Implement T-Learner using LGBMCLassifier
9) Implement X-Learner using XGBClassifier
10) Implement X-Learner using LGBMCLassifier
11) Outcome

![Data/Model_results.png](https://github.com/McGill-MMA-EnterpriseAnalytics/Airline-Passenger-Satisfaction-Prediction/blob/baddd7a9be720d85af68d4d04688f5d496ff9618/Data/Causal_results.png)

The learners' results are nearly identical, with Online_boarding showing the greatest ATE, trailed by Leg_room_service and Inflight_wifi_service. To boost customer satisfaction, the airline ought to concentrate on enhancing their performance in these areas of service.


#### Team Members:
Dreama Wang - 261112206

Nishi Nishi - 261078870

Riley Zhu - 261094733

ShanShan Lao - 261072808

Vibhu Bhardwaj - 261113187

Utkarsh Nagpal - 261071466

Mathilda Zhang - 261112212

Haoying Xu - 261109413



![bao-menglong--FhoJYnw-cg-unsplash](https://user-images.githubusercontent.com/47519737/219964307-0b876e94-6e03-4b4d-b31f-557d57b354dd.jpg)

