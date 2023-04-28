from airflow import DAG 
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
import time
import random
from airflow.utils.dates import days_ago
from datetime import datetime,timedelta
import time
import random
import os
import requests
import pandas as pd
from datetime import datetime
import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
import pickle
from sqlalchemy import create_engine
import ctgan
import numpy as np
import git

########
import seaborn as sns
from seaborn import diverging_palette
from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *
from airflow.operators.bash_operator import BashOperator

############

from sklearn.metrics import mean_absolute_error
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from sklearn.linear_model import (
    LogisticRegression,
    Lasso
)

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    RepeatedStratifiedKFold,
    GridSearchCV,
    KFold
)

from sklearn.metrics import (
    accuracy_score, 
    precision_score, average_precision_score,
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    classification_report
)

import lightgbm as lgb
from lightgbm import LGBMClassifier, plot_importance

from yellowbrick.classifier import (
    ConfusionMatrix,
    ROCAUC
)

#this should be usually encrypted
my_conninitial=create_engine("mysql+mysqldb://root:rootroot@host.docker.internal/airlinedatabase")
currentdate=str(datetime.now())


def load_data():
    my_conninitial=create_engine("mysql+mysqldb://root:rootroot@host.docker.internal/airlinedatabase")
    query='SELECT * FROM maindata'
    temptransferdf=pd.read_sql(query,my_conninitial)
    temptransferdf.fillna(value=pd.np.nan, inplace=True)
    Source='/opt/airflow/CaeData/Airlinetempdata/'
    temptransferdf.to_csv(Source+'temptransferdf.csv')
def split_data():
    Source='/opt/airflow/CaeData/Airlinetempdata/'
    
    query='SELECT * FROM maindata'
    temptransferdf=pd.read_sql(query,my_conninitial)
    temptransferdf.fillna(value=pd.np.nan, inplace=True)
    now = datetime.datetime.now()
    now = now.strftime("%m/%d/%Y, %H:%M:%S")
    temptransferdf.fillna(value=pd.np.nan, inplace=True)
    temptransferdf['TimeStamp'] = pd.to_datetime(temptransferdf['TimeStamp'])
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=60)
    first_30_days = temptransferdf[temptransferdf['TimeStamp'] >= cutoff_date]
    end_date = datetime.datetime.now() - datetime.timedelta(days=120)
    start_date = end_date - datetime.timedelta(days=60)
    mid_30_days = temptransferdf[(temptransferdf['TimeStamp'] >= start_date) & (temptransferdf['TimeStamp'] < end_date)]
    first_30_days.to_sql(con=my_conninitial,name='first_30_days',if_exists='overwrite',index=False)
    mid_30_days.to_sql(con=my_conninitial,name='mid_30_days',if_exists='overwrite',index=False)

def compare_data_drift():
    mid_30_days=pd=pd.read_sql('SELECT * FROM mid_30_days',my_conninitial)
    first_30_day=pd.read_sql('SELECT * FROM first_30_days',my_conninitial)   
    ref_data = mid_30_days
    pred_data = first_30_day
    cols_to_drop = ['Unnamed: 0', 'id','TimeStamp']
    pred_data = pred_data.drop(cols_to_drop, axis=1)
    ref_data = ref_data.drop(cols_to_drop, axis=1)
    column_mapping = {}
    column_mapping['target'] = 'satisfaction'
    column_mapping['prediction'] = None
    column_mapping['datetime'] = None

    column_mapping['numerical_features'] = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Inflight entertainment', 'Food and drink','Seat comfort',
                                            'On-board service', 'Leg room service', 'Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes','Online boarding']
    column_mapping['categorical_features'] = ['Gender', 'Customer Type','Type of Travel','Class']
    tests = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=ref_data, current_data=pred_data)
    tests
    selected_columns = ['Inflight wifi service', 'Checkin service', 'Seat comfort', 'Flight Distance','Customer Type','Departure/Arrival time convenient','satisfaction']
    ref_data = ref_data.loc[:, selected_columns]
    pred_data = pred_data.loc[:, selected_columns]
    report = Report(metrics=[
        DataDriftPreset(), 
    ])

    report.run(reference_data=ref_data, current_data=pred_data)
    report
    data = report.as_dict()
    Ndriftedcolumns=0
    for metric in data['metrics']:
        if metric['metric'] == 'DataDriftTable':
            drift_by_columns = metric['result']['drift_by_columns']
            for column in drift_by_columns:
                column_data = drift_by_columns[column]
                column_name = column_data['column_name']
                column_type = column_data['column_type']
                drift_detected = column_data['drift_detected']
                print(f"Column name: {column_name}, Type: {column_type}, Drift detected: {drift_detected}")
                n=n+1


    if Ndriftedcolumns>0:
        raise Exception
    

def retrain_model_task():
    my_conninitial=create_engine("mysql+mysqldb://root:rootroot@host.docker.internal/airlinedatabase")
    query='SELECT * FROM maindata'
    temptransferdf=pd.read_sql(query,my_conninitial)
    temptransferdf.fillna(value=pd.np.nan, inplace=True)
    data=temptransferdf.copy()

    def clf_score(clf, X_train, y_train, X_val, y_val, train=True):
        if train:
            print("Train Result:\n")
            print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
            print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
            print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))
            res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
            print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
            print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        elif train == False:
            print("Validation Result:\n")
            print("accuracy score: {0:.4f}\n".format(accuracy_score(y_val, clf.predict(X_val))))
            precision, recall, _ = precision_recall_curve(y_val, clf.predict(X_val))
            average_precision = average_precision_score(y_val, clf.predict(X_val))
            fpr, tpr, _ = roc_curve(y_val, clf.predict(X_val))
            roc_auc = roc_auc_score(y_val, clf.predict(X_val))
            print("roc auc score: {}\n".format(roc_auc))        
            print("Classification Report: \n {}\n".format(classification_report(y_val, clf.predict(X_val))))
            print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_val, clf.predict(X_val))))
            #plot_confusion_matrix(clf,  X_val, clf.predict(X_val))
            print("End of validation Result\n")


    def list_column_values(df, number_of_values, print_all):
        for col in df.columns[0:]:
            if df[col].nunique() <= number_of_values:
                print(f"{col.ljust(25)}" +  ' ==> ' + str(df[col].sort_values().unique().tolist()) )
            else:  
                if print_all=='True':
                 print(f"{col.ljust(25)}" + ' ==> more than ' + str(number_of_values) + ' values')



    numeric_cols = ['Age', 'Flight_Distance','Departure_Delay_in_Minutes']
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    #Handling Missing values
    list_column_values(data, data.shape[1],'True')
    incomplete = ['Inflight_wifi_service','Departure/Arrival_time_convenient',
            'Ease_of_Online_booking','Online_boarding','Leg_room_service']
    data = data.loc[(data[incomplete] != 0).all(axis=1)]

    X = data.drop(columns=['satisfaction'])
    y = data['satisfaction']

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size = 0.2, random_state = 777)
    N_train, _ = X_train.shape 
    N_val_test,  _ = X_val_test.shape 
    N_train, N_val_test

    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = 777)

    N_val, _ = X_val.shape 
    N_test,  _ = X_test.shape 

    N_val, N_test

    lgbm_clf = LGBMClassifier(n_estimators=1000, 
                            n_jobs=1,
                            learning_rate=0.02,
                            num_leaves=50,
                            max_depth=7,
                            verbose=-1,
                            random_state=42)
    model_lgbm= lgbm_clf.fit(X_train, y_train)
    clf_score(model_lgbm, X_train, y_train, X_val, y_val, train=False)
    model_lgbm
    Filename="model_lgbm"+currentdate+".pkl"
    Destination='/opt/airflow/CaeData/ModelRepo/'+Filename
    pickle.dump(model_lgbm, open(Destination, 'wb'))


def load_final_model():
    Filename="model_lgbm"+currentdate+".pkl"
    Destination='/opt/airflow/CaeData/ModelRepo/'+Filename
    # repo_path = '/path/to/local/folder/repository'
    # file_path = Destination
    # commit_message = 'Added my_file.pkl'
    # repo = git.Repo(repo_path)
    # repo.index.add([file_path])
    # repo.index.commit(commit_message)
    # origin = repo.remote(name='origin')
    # origin.push()
    # print('File pushed to GitHub successfully!')



def LogStart():
    print(datetime.datetime.today())

default_args={
    'retry':5,
    'retry_delay':timedelta(minutes=1)
    }


with DAG(dag_id="2DriftinData",default_args=default_args,schedule_interval='@daily',start_date=days_ago(3),catchup=False) as dag:
    # Waititng_for_sensor= FileSensor(
    #     task_id='Waititng_for_sensor',
    #     #python_callable=_downloading_data
    # )
    
    LogStart= PythonOperator(
        task_id='LogStart',
        python_callable=LogStart
    )

    load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    )

    split_data = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    )

    compare_data_drift = PythonOperator(
    task_id='compare_data_drift',
    python_callable=compare_data_drift,
    )

    retrain_model_task = PythonOperator(
    task_id='retrain_model_task',
    python_callable=retrain_model_task,
    )
    load_final_model = PythonOperator(
    task_id='load_final_model',
    python_callable=load_final_model,
    )

# To fetch code to implement CICD as future scope
    # t1 = BashOperator(
    # task_id='pull_script',
    # bash_command='curl -o /local/folder/my_script.py https://raw.githubusercontent.com/<username>/<repository>/<branch>/<path_to_file>',
    # )

    
LogStart>>load_data>>split_data>>compare_data_drift>>retrain_model_task>>load_final_model

    
