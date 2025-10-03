#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
INPUT_FILE_PATTERN = os.getenv("INPUT_FILE_PATTERN")
OUTPUT_FILE_PATTERN = os.getenv("OUTPUT_FILE_PATTERN")

print("S3 endpoint:", S3_ENDPOINT_URL)
print("Input pattern:", INPUT_FILE_PATTERN)
print("Output pattern:", OUTPUT_FILE_PATTERN)


def read_data(path: str):
    if S3_ENDPOINT_URL:
        storage_options = {
            "key": "test",
            "secret": "test",
            "client_kwargs": {
                "endpoint_url": S3_ENDPOINT_URL
            }
        }
        print(f"reading from S3 path: {path} (endpoint={S3_ENDPOINT_URL})")
        return pd.read_parquet(path, storage_options=storage_options)

    return pd.read_parquet(path)

def save_data(df: pd.DataFrame, path: str):
    storage_options = None
    if S3_ENDPOINT_URL:
        storage_options = {
            "key": "test",
            "secret": "test",
            "client_kwargs": {
                "endpoint_url": S3_ENDPOINT_URL
            }
        }

    df.to_parquet(
        path,
        engine="pyarrow",
        index=False,
        compression=None,
        storage_options=storage_options
    )

    print(f"output has been saved into {path}")

def prepare_data (df: pd.DataFrame, categorical=None):
    df = df.copy()

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    if categorical:
        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = INPUT_FILE_PATTERN or default_input_pattern
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 'output/taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
    output_pattern = OUTPUT_FILE_PATTERN or default_output_pattern
    return output_pattern.format(year=year, month=month)

def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']

    # input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    # output_file = f'output/taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

    input_file = get_input_path(year, month)
    print(f'input_file: {input_file}')
    output_file = get_output_path(year, month)
    print(f'output_file: {output_file}')

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file)
    print(f'df count: {len(df)}')
    df = prepare_data(df, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    total_predicted_duration = y_pred.sum()
    print("sum of predicted durations:", total_predicted_duration)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)

if __name__ == "__main__":

    try:
        os.makedirs("./output", exist_ok=True)

        year = int(sys.argv[1])
        month = int(sys.argv[2])

        main(year, month)
    except Exception as e:
        print("Error:", e)
