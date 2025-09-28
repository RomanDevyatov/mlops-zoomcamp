import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from prefect import task, flow
from prefect.logging import get_run_logger

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnQuantileMetric, ColumnValueRangeMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists metrics_homework;
create table metrics_homework(
	timestamp TIMESTAMP,
	fare_amount_p50 FLOAT,
	number_of_values INTEGER
)
"""

# reference_data = pd.read_parquet('data/reference.parquet')
# with open('models/lin_reg.bin', 'rb') as f_in:
# 	model = joblib.load(f_in)

raw_data = pd.read_parquet('data/green_tripdata_2024-03.parquet')
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(
			metrics=[
				ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
				ColumnValueRangeMetric(column_name="fare_amount", left=0, right=200),
			]
)

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=admin", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=admin") as conn:
			conn.execute(create_table_statement)

@task(cache_policy=None)
def calculate_metrics_postgresql(curr, i):
	logger = get_run_logger()
	begin = pd.Timestamp("2024-03-01")

	current_data = raw_data[
		(raw_data.lpep_pickup_datetime >= (begin + pd.Timedelta(days=i)))
		& (raw_data.lpep_pickup_datetime < (begin + pd.Timedelta(days=i + 1)))
	]

	logger.info(f"current_data len: {len(current_data)}")

	if current_data.empty:
		median_fare = None
		number_of_values = None
	else:

		report.run(reference_data=None, current_data=current_data, column_mapping=column_mapping)
		result = report.as_dict()

		median_fare = result["metrics"][0]["result"]["current"]["value"]
		number_of_values = result["metrics"][1]["result"]["current"]["number_of_values"]

	logger.info(f"running sql. median_fare: {median_fare}, number_of_values: {number_of_values}")
	curr.execute(
		"INSERT INTO metrics_homework(timestamp, fare_amount_p50, number_of_values) values (%s, %s, %s)",
		(begin + datetime.timedelta(i), median_fare, number_of_values)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=admin", autocommit=True) as conn:
		for i in range(0, 31):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()
