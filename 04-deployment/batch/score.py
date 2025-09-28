#!/usr/bin/env python
# coding: utf-8
import os
import uuid
import pickle

import pandas as pd

import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from pathlib import Path

MLFLOW_TRACKING_URI = 'http://localhost:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def run():
    RUN_ID = 'e27e6b42b49b43eebdd1e58f9c4f1c95'
    year = 2021
    month = 2
    taxi_type = 'green'
    
    # input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    input_file = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet'
    output_file = f'output/green/green_tripdata_2021-01.parquet'
    
    input_file2 = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet'
    output_file2 = f'output/green/green_tripdata_2021-02.parquet'
    
    inputs = {
        input_file: output_file,
        input_file2: output_file2
    }
    
    os.makedirs("output/green", exist_ok=True)


# In[9]:


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df['ride_id'] = generate_uuids(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# In[12]:


def load_model(run_id):
    # logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):

    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    
    model = load_model(run_id)
    y_pred = model.predict(dicts)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    df_result.to_parquet(output_file, index=False)


# In[37]:


for key, value in inputs.items():
    print(f"input: {key}, {value}")
    apply_model(input_file=key, run_id=RUN_ID, output_file=value)


# In[38]:


get_ipython().system('ls output/green/')


# In[22]:


ls 


# In[35]:


import shutil

shutil.rmtree("output")


# In[ ]:





# In[ ]:




