from functools import partial
import sys
import time
import os
from os.path import dirname, join, realpath

import math
import numpy as np

# ndarray for type hints
from numpy import ndarray
from numpy import nan
import pandas as pd
from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
import matplotlib.pyplot as plt
import schedule
import yaml
import random
import snoop
import asyncio
import requests
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

# import datetime
from datetime import datetime, date, timedelta
from requests.exceptions import Timeout, ConnectionError

# In order to run this script directly, you need to add the parent directory to the sys.path
# Or you need to run this script in the parent directory using the command: python -m client.algorithm
sys.path.append(dirname(realpath(".")))
from common.utility import get_latency, get_latency_sync
from common.config_logging import init_logging

logger = init_logging(
    join(dirname(realpath(__file__)), "network_test_visualization.log")
)

# read config.yml file using yaml
with open(
    join(dirname(realpath(__file__)), "config.yml"), mode="r", encoding="utf-8"
) as file:
    config = yaml.safe_load(file)

logger.info(f"load config: {config}")

# 分组粒度，单位秒
group_size: int = config["network_test_visualization"]["group_size"]
# 计算的百分位数值
percentage_values: list[int] = config["network_test_visualization"]["percentage_values"]


def p50(x) -> float:
    return partial(np.percentile, q=50, method="nearest")(x)


def p99(x) -> float:
    return partial(np.percentile, q=99, method="nearest")(x)


def calculation(input_file_path: str, output_identifier= '') -> DataFrame:
    df: DataFrame = pd.read_csv(input_file_path)
    # get the column names of the df
    cloud_ids: list[str] = df.columns.values[1:].tolist()
    # group by timestamp in group_size and calculate the percentage latency
    # make the timestamp_group equal to the upper value of group_size of the timestamp
    # eg: group_size = 60s(1 minute), timestamp = 2022-11-02 00:25:15 then timestamp_group = 2022-11-02 00:26:00 in seconds since epoch for performance
    def timestamp_group(x: str) -> int:
        timestamp = datetime.strptime(
            x["timestamp"], "%Y-%m-%d %H:%M:%S.%f"
        ).timestamp()
        return int(timestamp - timestamp % group_size + group_size)

    df["timestamp_group"] = df.apply(timestamp_group, axis=1)
    df_timestamp_group: DataFrameGroupBy = df.groupby("timestamp_group")
    # calculate the aggregated df
    df_aggregated = df_timestamp_group.agg(
        {cloud_ids[0]: ["min", "max", "mean", p50, p99]}
    )
    # rename the multiindex columns to plain columns
    df_aggregated_columns = list(map("_".join, df_aggregated.columns))
    for cloud_id in cloud_ids[1:]:
        df_aggregated_temp = df_timestamp_group.agg(
            {cloud_id: ["min", "max", "mean", p50, p99]}
        )
        df_aggregated_columns += list(map("_".join, df_aggregated_temp.columns))
        df_aggregated = pd.concat(
            [df_aggregated, df_aggregated_temp], ignore_index=True, axis=1
        )

    print(f'df_aggregated.shape: {df_aggregated.shape}')
    df_aggregated.columns = df_aggregated_columns
    # save the aggregated df to csv file
    csv_file_path = join(dirname(realpath(__file__)), "network_test_results", f"network_test_aggregated_{output_identifier}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
    df_aggregated.to_csv(csv_file_path)
    return df_aggregated

def simple_visualization_from_csv(input_csv_file: str):
    df = pd.read_csv(input_csv_file)
    simple_visualization_from_dataframe(df)

def simple_visualization_from_dataframe(df: DataFrame):
    x = list(map(lambda x: datetime.fromtimestamp(x).strftime('%Y_%m_%d_%H_%M_%S'), df.index))
    plt.scatter(x, df['cloud_id_0_p99'], c ="green")
    plt.scatter(x, df['cloud_id_0_p50'], c ="blue")
    
    plt.scatter(x, df['cloud_id_1_p99'], c ="red")
    plt.scatter(x, df['cloud_id_1_p50'], c ="purple")
    
    plt.scatter(x, df['cloud_id_2_p99'], c ="black")
    plt.scatter(x, df['cloud_id_2_p50'], c ="yellow")
    
    plt.scatter(x, df['cloud_id_3_p99'], c ="gray")
    plt.scatter(x, df['cloud_id_3_p50'], c ="orange")
    
    plt.xlabel("timestamp")
    plt.ylabel("latency/s")
    plt.show()

if __name__ == "__main__":
    df = calculation(
        join(
            dirname(realpath(__file__)),
            "network_test_results",
            "network_test_with_placements_1_1_1_1_datasize_10485760_write_start_at_2022_02_11_00_25_15.csv",
        )
    )
    simple_visualization_from_dataframe(df)
    # simple_visualization_from_csv('network_test_results/network_test_aggregated__2022_11_02_16_29_28.csv')
    # simple_visualization_from_csv('network_test_results/network_test_aggregated__2022_11_02_17_05_43.csv')
