from pathlib import Path
import pandas as pd

from datetime import datetime

import homework.batch as batch_module

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    actual_df = batch_module.prepare_data(df, categorical=['PULocationID', 'DOLocationID'])

    expected_data = [
        {"PULocationID": "-1", "DOLocationID": "-1",
         "tpep_pickup_datetime": dt(1, 1), "tpep_dropoff_datetime": dt(1, 10),
         "duration": 9.0},
        {"PULocationID": "1", "DOLocationID": "1",
         "tpep_pickup_datetime": dt(1, 2), "tpep_dropoff_datetime": dt(1, 10),
         "duration": 8.0},
    ]

    expected_df = pd.DataFrame(expected_data)

    actual_dict = actual_df.to_dict(orient="records")
    expected_dict = expected_df.to_dict(orient="records")

    print(f"expected count: {len(expected_dict)}")

    assert actual_dict == expected_dict


