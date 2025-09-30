#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os

def reading(filename: str):
    return pd.read_parquet(filename)

def prepare_data (df: pd.DataFrame, categorical=None):
    df = df.copy()

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    if categorical:
        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def read_data(filename, categorical):
    df = reading(filename)
    
    prepare_data (df, categorical)
    
    return df

def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    # output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred


    df_result.to_parquet(output_file, engine='pyarrow', index=False)

if __name__ == "__main__":
    try:
        os.makedirs("./output", exist_ok=True)

        year = int(sys.argv[1])
        month = int(sys.argv[2])

        main(year, month)
    except Exception as e:
        print(e)
