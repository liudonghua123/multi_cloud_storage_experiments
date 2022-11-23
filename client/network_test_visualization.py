#!/usr/bin/env python

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
import mplcursors
import yaml
import random
import snoop
import asyncio
import requests
import fire
from os.path import exists
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

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

# read config-default.yml, config.yml(optional, override the default configurations) using yaml
default_config_file = "config-default.yml"
config_file = "config.yml"
logger.info(f"Loading default config file: {config_file}")
with open(join(dirname(realpath(__file__)), default_config_file), mode="r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
if os.path.exists(join(dirname(realpath(__file__)), config_file)):
    logger.info(f"Loading override config file: {config_file}")
    with open(join(dirname(realpath(__file__)), config_file), mode="r", encoding="utf-8") as file:
        # use **kwargs to merge the two dictionaries
        config = {**config, **yaml.safe_load(file)}
logger.info(f"load config: {config}")

# 分组粒度，单位秒
group_size: int = config["network_test_visualization"]["group_size"]
# 计算的百分位数值
percentage_values: list[int] = config["network_test_visualization"]["percentage_values"]
# x轴刻度数量，即x轴上的刻度点数量，由于x轴是时间很长，显示效果不理想，所以需要缩小x轴的刻度点数量
# 使用 range(start_timestamp, end_timestamp, step) 来生成x轴刻度点
# step=int(timestamp_size / xtick_size), 并且需要接近于 group_size 的整数倍
xtick_size: int = config["network_test_visualization"]["xtick_size"]

# 美化图表，使用 seaborn styles
# MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
plt.style.use('seaborn')
# plt.style.use('seaborn-whitegrid')

def p50(x) -> float:
    return partial(np.percentile, q=50, method="nearest")(x)


def p90(x) -> float:
    return partial(np.percentile, q=90, method="nearest")(x)


def p99(x) -> float:
    return partial(np.percentile, q=99, method="nearest")(x)


def calculation(input_file_path: str, output_identifier="") -> DataFrame:
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
        {cloud_ids[0]: ["min", "max", "mean", p50, p90, p99]}
    )
    logger.debug(f"df_aggregated: {df_aggregated}, df_aggregated.index: {df_aggregated.index}, df_aggregated.columns: {df_aggregated.columns}")
    for cloud_id in cloud_ids[1:]:
        df_aggregated = pd.concat(
            [
                df_aggregated, 
                df_timestamp_group.agg(
                    {cloud_id: ["min", "max", "mean", p50, p90, p99]}
                )
            ],
            axis=1,
        )
    # rename the multiindex columns to plain columns
    logger.debug(f"df_aggregated.shape: {df_aggregated.shape}, df_aggregated.columns: {df_aggregated.columns}")
    df_aggregated.columns = map("_".join, df_aggregated.columns)
    logger.debug(f"rename the multiindex columns to plain columns, df_aggregated.columns: {df_aggregated.columns}")
    # save the aggregated df to csv file
    csv_file_path = join(
        dirname(realpath(__file__)),
        "network_test_results",
        f"network_test_aggregated_{output_identifier}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
    )
    # index of df_aggregated is timestamp_group here.
    # DataFrame.to_csv default to save with index and with header, so no problem here.
    logger.info(f'write df_aggregated to csv file {csv_file_path}')
    df_aggregated.to_csv(csv_file_path)
    return df_aggregated


def simple_visualization_from_csv(
    input_csv_file: str,
    cloud_ids,
    aggregations,
    legend_loc,
):
    # read the aggregated csv file, default index_col is None for read_csv,
    # and should specify the index_col to be timestamp_group
    # or the df will not be the same structure as the one returned by calculation function.
    # df.index is timestamp_group column now
    df = pd.read_csv(input_csv_file, index_col="timestamp_group")
    simple_visualization_from_dataframe(df, cloud_ids, aggregations, legend_loc)


def simple_visualization_from_dataframe(
    df: DataFrame,
    cloud_ids,
    aggregations,
    legend_loc,
):
    logger.info(f"df.shape: {df.shape}, cloud_ids: {cloud_ids}, aggregations: {aggregations}")
    # df.index is timestamp_group column here
    # x = list(map(lambda x: datetime.fromtimestamp(x).strftime("%H_%M"), df.index))
    x = df.index.tolist()
    # parse cloud_ids and aggregations
    cloud_id_list_supported = set(map(lambda x: x[: x.rindex("_")], df.columns))
    # the fire module will parse the string to list, so we do not need to parse it here
    # cloud_ids = cloud_ids.split(",")
    # aggregations = aggregations.split(",")
    # check cloud_ids, should included in df.columns
    if not set(cloud_ids) <= cloud_id_list_supported:
        print(f"cloud_ids: {cloud_ids} should be included in {cloud_id_list_supported}")
        exit(-1)
    # check aggregations, should be included in ['min', 'max', 'mean', 'p50', 'p90', 'p99']
    aggregation_list_supported = set(["min", "max", "mean", "p50", "p90", "p99"])
    if not set(aggregations) <= aggregation_list_supported:
        print(
            f"aggregations: {aggregations} should be included in {aggregation_list_supported}, only these aggregations are supported."
        )
        exit(-1)

    # use plt.scatter to plot timestamp_group as X axis
    # and the matrix data specified by cloud_ids and aggregations as Y axis
    column_combinations = list(
        map(
            lambda item: f"{item[0]}_{item[1]}",
            product(cloud_ids, aggregations),
        )
    )
    # Create subplot
    fig, ax = plt.subplots()
    fig.suptitle('network_test_visualization', fontweight ="bold")
    for column_combination in column_combinations:
        ax.scatter(x, df[column_combination] * 1000)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("latency/ms")
    step = int((x[-1] - x[0]) / xtick_size)
    logger.debug(f"x[0]: {x[0]}, x[-1]: {x[-1]}, step: {step}")
    step = int(step / group_size) * group_size
    logger.debug(f"corrective step: {step}")
    x_pos = np.arange(x[0], x[-1], step)
    logger.debug(f"xtick_size: {xtick_size}, len(x_pos): {len(x_pos)}, x_pos: {x_pos}")
    x_pos_label = list(map(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M"), x_pos))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_pos_label)
    # ax.set_xticklabels(x_pos_label, rotation=45, ha='right')
    fig.autofmt_xdate(rotation=45)
    ax.legend(column_combinations, loc=legend_loc)
    # Enable add some annotation when mouse hover on the point 
    # mplcursors.cursor(ax,hover=True) # only show Y value, why?
    mplcursors.cursor(ax,hover=True).connect("add", lambda sel: sel.annotation.set_text(f"x: {datetime.fromtimestamp(sel.target_[0]).strftime('%Y-%m-%d %H:%M')}\ny: {sel.target_[1]}"))
    plt.show()


def main(
    network_test_results_csv_file_path: str = None,
    network_test_aggregated_results_csv_file_path: str = None,
    cloud_ids: str = "cloud_id_0,cloud_id_1,cloud_id_2,cloud_id_3".split(","),
    aggregations: str = "p50,p99".split(","),
    legend_loc: str = "upper right",
):
    """
    Visualize the network test aggregated results.

    If network_test_aggregated_results_csv_file_path provided, then use the date of csv file to visualize.
    Else, use the network_test_results_csv_file_path to calculate the aggregated results and visualize.

    Parameters:
    network_test_results_csv_file_path (str): the csv file path of network test results.
    network_test_aggregated_results_csv_file_path (str): the csv file path of network test aggregated results.
    cloud_ids (str): the cloud ids to visualize, separated by comma.
    aggregations (str): the aggregations to visualize, separated by comma.
    legend_loc (str): the location of legend, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html for more details.

    Returns:
    0: if success
    -1: if failed

    """
    # print the arguments
    logger.info(f'arguments passed are network_test_results_csv_file_path: {network_test_results_csv_file_path}, network_test_aggregated_results_csv_file_path: {network_test_aggregated_results_csv_file_path}, cloud_ids: {cloud_ids}, aggregations: {aggregations}, legend_loc: {legend_loc}')
    # use network_test_aggregated_results_csv_file_path priority
    if network_test_aggregated_results_csv_file_path != None:
        if not exists(network_test_aggregated_results_csv_file_path):
            print(
                f"network_test_aggregated_results_csv_file_path: {network_test_aggregated_results_csv_file_path} not exists"
            )
            print(
                f"please check the network_test_aggregated_results_csv_file_path or use network_test_results_csv_file_path"
            )
            exit(-1)
        logger.info(f'network_test_aggregated_results_csv_file_path is used')
        simple_visualization_from_csv(
            network_test_aggregated_results_csv_file_path,
            cloud_ids,
            aggregations,
            legend_loc,
        )
        return 0
    if network_test_results_csv_file_path == None:
        print(
            "Either network_test_results_csv_file_path or network_test_aggregated_results_csv_file_path should be specified"
        )
        exit(-1)
    elif not exists(network_test_results_csv_file_path):
        print(
            f"network_test_results_csv_file_path: {network_test_results_csv_file_path} not exists"
        )
        exit(-1)
    logger.info(f'network_test_results_csv_file_path is used')
    df = calculation(network_test_results_csv_file_path)
    simple_visualization_from_dataframe(df, cloud_ids, aggregations, legend_loc)
    return 0


if __name__ == "__main__":
    # Make Python Fire not use a pager when it prints a help text
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
