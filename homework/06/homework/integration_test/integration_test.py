# pylint: disable=duplicate-code

import pandas as pd
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
INPUT_FILE_PATTERN = os.getenv("INPUT_FILE_PATTERN")
print("S3_ENDPOINT_URL:", S3_ENDPOINT_URL)
print("INPUT_FILE_PATTERN:", INPUT_FILE_PATTERN)

if not S3_ENDPOINT_URL or not INPUT_FILE_PATTERN:
    raise ValueError("S3_ENDPOINT_URL и INPUT_FILE_PATTERN should be set!")

options = {
    "client_kwargs": {
        "endpoint_url": S3_ENDPOINT_URL,
        "region_name": "us-east-1"
    },
    "key": "test",
    "secret": "test"
}

input_file = INPUT_FILE_PATTERN.format(year=2023, month=1)

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

print(f"File has been saved into {input_file}")
