#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from pathlib import Path
from sklearn.linear_model import LinearRegression
import requests

import prefect
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from datetime import timedelta

EXPERIMENT_NAME = "nyc-taxi-experiment-hw-03"
DATA_FOLDER = 'data'
MODEL_NAME = "linear_regression_model"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def download_file(url: str, filename: str):
    """Download a file from a URL to a local path."""
    data_folder = Path(DATA_FOLDER)
    data_folder.mkdir(exist_ok=True)
    local_path = data_folder / filename

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return local_path

@task(retries=3, retry_delay_seconds=10, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def read_and_split_dataframe(year, month, val_size=0.2, random_state=42):
    logger = get_run_logger()
    
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
    try:
        logger.info(f"Downloading data from {url}")  
        local_file = download_file(url, f"yellow_tripdata_{year}-{month:02d}.parquet")
        logger.info(f"Saved to {local_file}")  
        
        logger.info(f"Reading from {local_file}")        
        df = pd.read_parquet(local_file)
        logger.info(f"Number of records in the dataframe: {len(df)}")

        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
    
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        logger.info(f"Dataframe size after filtering: {len(df)}")
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)

        logger.info(f"Splitting data into train and val sets..")
        n_val = int(len(df) * val_size)
        df_train = df.iloc[:-n_val]
        df_val = df.iloc[-100:]
        logger.info(f"len(df_train)={len(df_train)}, len(df_val)={len(df_val)}")
        
        return df_train, df_val

    except Exception as e:
        logger.error(f"Failed to read or process parquet file: {e}")
        raise

@task
def createX(df, dv=None): 
    logger = get_run_logger()
    
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    
    if len(df) == 0:
        logger.warning("Warning: Empty dataframe received in createX")
        raise ValueError("Empty dataframe")
    
    try:
        dicts = df[categorical + numerical].to_dict(orient='records')
        logger.info(f"len(dicts)={len(dicts)}")

        if dv is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)

        return X, dv
    except Exception as e:
        logger.error(f"Failed in createX: {e}")
        raise

@task
def train_model(X_train, y_train, X_val, y_val, dv):
    logger = get_run_logger()
    logger.info("Starting model training")
    
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        logger.info(f"Model trained. Intercept: {lr.intercept_}")
        mlflow.log_param("intercept", lr.intercept_)
        
        y_pred = lr.predict(X_val)        
        rmse = root_mean_squared_error(y_val, y_pred)
        logger.info(f"Validation RMSE: {rmse}")
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
            
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        input_example = pd.DataFrame(X_train[:5].toarray() if hasattr(X_train, "toarray") else X_train)

        mlflow.sklearn.log_model(lr, name=MODEL_NAME, input_example=input_example)

        return run.info.run_id

@task
def run_register_model(run_id):
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/{MODEL_NAME}",
        name="LinearRegression"
    )


@flow(name="NYC Taxi ML Pipeline")
def run_ml_pipeline(year, month):
    try:
        logger = get_run_logger()
        logger.info(f"Starting ML pipeline. Prefect version: {prefect.__version__}")
        logger.info(f"Data loading and splitting..")
        df_train, df_val = read_and_split_dataframe(year, month)

        logger.info(f"Feature engineering (X_train)..")
        X_train, dv = createX(df_train)
        logger.info(f"Feature engineering (X_val)..")
        X_val, _ = createX(df_val, dv)

        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values

        logger.info(f"Training..")
        run_id = train_model(X_train, y_train, X_val, y_val, dv)

        logger.info(f"MLflow run_id: {run_id}")

        logger.info(f"Registering the model...")
        run_register_model(run_id)

        logger.info(f"Saving the run_id locally...")
        with open(models_folder / "last_run_id.txt", "w") as f:
            f.write(run_id)
            
    except Exception as e:
        logger.error(f"Failed in run_ml_pipeline: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a duration prediction model")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=3)
    args = parser.parse_args()
    
    run_ml_pipeline(args.year, args.month)
   
