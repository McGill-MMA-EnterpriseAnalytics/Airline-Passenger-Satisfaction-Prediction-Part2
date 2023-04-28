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


def GenerateSamples():
    my_conninitial=create_engine("mysql+mysqldb://root:rootroot@host.docker.internal/airlinedatabase")
    Source='/opt/airflow/CaeData/'
    pickled_model = pickle.load(open(Source+"Datagenerator.pkl", 'rb'))
    newsamples = pickled_model.sample(100)
    my_datetime = datetime.datetime.now()
    # query='SELECT * FROM '+table
    # temptransferdf=pd.read_sql(query,my_connwarehouse)
    newsamples["TimeStamp"] = my_datetime
    newsamples.to_sql(con=my_conninitial,name='maindata',if_exists='append',index=False)


def LogStart():
    print(datetime.datetime.today())

    time.sleep(random.randint(1,1))

default_args={
    'retry':5,
    'retry_delay':timedelta(minutes=1)
    }


with DAG(dag_id="1SyntheticData",default_args=default_args,schedule_interval='@daily',start_date=days_ago(3),catchup=False) as dag:
    # Waititng_for_sensor= FileSensor(
    #     task_id='Waititng_for_sensor',
    #     #python_callable=_downloading_data
    # )


    
    LogStart= PythonOperator(
        task_id='LogStart',
        python_callable=LogStart
    )

    GenerateSamples = PythonOperator(
    task_id='GenerateSamples',
    python_callable=GenerateSamples,
    )


    

    LogStart>>GenerateSamples

    
