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
from common.config_logging import init_logging

logger = init_logging(
    join(dirname(realpath(__file__)), "algorithm_visualization.log")
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

# # 算法结果可视化
# algorithm_visualization:
#   # 比较的算法
#   algorithms:
#     - aw_cucb
#     - simple
#     - ewma
#   # 算法比较指标
#   metrics:
#     - post_reward_accumulated_average
#     - post_cost_accumulated_average
#     - latency
#   # 是否显示为子图
#   subplot: false
#   # 是否统计节点请求数据
#   node_statistics: true
#   # 扩展指标, 不直接参与比较, 一般需要处理后才能使用
#   extra_metrics: 
#     - placement_policy

# 比较的算法
algorithms: list[str] = config["algorithm_visualization"]["algorithms"]
# 计算的百分位数值
metrics: list[str] = config["algorithm_visualization"]["metrics"]
# 是否显示为子图
subplot: bool = config["algorithm_visualization"]["subplot"]
# 是否统计节点请求数据
node_statistics: bool = config["algorithm_visualization"]["node_statistics"]
# 扩展指标
extra_metrics: list[str] = config["algorithm_visualization"]["extra_metrics"]

# 美化图表，使用 seaborn styles
# MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
plt.style.use('seaborn')
# plt.style.use('seaborn-whitegrid')

def plot_placement_policy(ax, df, algorithms, legend_loc):
  statistics_results = { algorithm: {} for algorithm in algorithms }
  # statistics the node frequency
  for algorithm in algorithms:
    # iterate the f'placement_policy_{algorithm}' column
    for data in df[f'placement_policy_{algorithm}']:
      nodes = map(int, data.split('_'))
      for node in nodes:
        if node not in statistics_results[algorithm]:
          statistics_results[algorithm][node] = 0
        statistics_results[algorithm][node] += 1
  logger.info(f'statistics_results: {statistics_results}')
  # prepare the data for plot
  # node_keys = [0, 1, ...]
  nodes_keys = statistics_results[algorithms[0]].keys()
  nodes_keys = sorted(nodes_keys)
  # plot the node frequency bar
  for i, algorithm in enumerate(algorithms):
    x = list(map(lambda x: x + i * 0.2, nodes_keys))
    y = [statistics_results[algorithm][node] for node in nodes_keys]
    logger.info(f'algorithm: {algorithm}, x: {x}, y: {y}')
    ax.bar(x, y, width = 0.1, label=algorithm)
    ax.legend(loc=legend_loc, shadow=True)
  mplcursors.cursor(ax, hover=True)
  plt.show()

def visualization_from_dataframe(
    df: DataFrame,
    algorithms: list[str],
    metrics: list[str],
    subplot: bool,
    legend_loc: str,
):
  logger.info(f'''visualization_from_dataframe df.shape: {df.shape}, 
    df.columns: {df.columns}, len(df): {len(df)}, df.head(): {df.head()}''')

  # Create subplot for each metric
  if subplot:
    def plot_metric(ax, metric, df, algorithms, legend_loc):
      for algorithm in algorithms:
        x = df["tick"]
        y = df[f'{metric}_{algorithm}']
        ax.plot(x, y, label=f'{algorithm}', mouseover=True)
        ax.set_title(metric, fontweight="bold")
        ax.legend(loc=legend_loc, shadow=True)
      mplcursors.cursor(ax, hover=True)
    fig, axes = plt.subplots(len(metrics + extra_metrics), 1)
    fig.tight_layout()
    # for metrics
    for ax, metric in zip(axes[:len(metrics)], metrics):
      plot_metric(ax, metric, df, algorithms, legend_loc)
    # for extra_metrics
    for ax, metric in zip(axes[len(metrics):], extra_metrics):
      if metric == "placement_policy":
        plot_placement_policy(ax, df, algorithms, legend_loc)
    plt.show()
  else:
    def plot_metric(metric, df, algorithms, legend_loc):
      plt.figure()
      for algorithm in algorithms:
        x = df["tick"]
        y = df[f'{metric}_{algorithm}']
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        plt.plot(x, y, label=f'{algorithm}', mouseover=True)
      plt.title(metric, fontweight="bold")
      # Adding legend, which helps us recognize the curve according to it's color
      plt.legend(loc=legend_loc, shadow=True)
      mplcursors.cursor(hover=True)
      plt.show()
      
    for metric in metrics:
      plot_metric(metric, df, algorithms, legend_loc)
    
    # TODO: multi-threading not working here, the line is not shown! why?
    # UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
    # with ThreadPoolExecutor(max_workers=len(metrics)) as executor:
    #   size_of_metrics = len(metrics)
    #   executor.map(plot_metric, metrics, [df] * size_of_metrics, [algorithms] * size_of_metrics, [legend_loc] * size_of_metrics)


def main(
    input_dir: str = join(dirname(__file__), "results_processed_test"),
    algorithms: list[str] = algorithms,
    metrics: list[str] = metrics,
    subplot: bool = subplot,
    legend_loc: str = "upper right",
):
  """
  Visualize the different algorithms' results.

  Parameters:
  algorithms (list[str]): the algorithms to visualize, separated by comma.
  metrics (list[str]): the metrics of the algorithms to visualize, separated by comma.
  subplot (bool): use subplot to visualize the results or not.
  legend_loc (str): the location of legend, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html for more details.


  Returns:
  0: if success
  -1: if failed

  """
  # print the arguments
  logger.info(
    f'arguments passed are input_dir: {input_dir}, algorithms: {algorithms}, metrics: {metrics}, subplot: {subplot}, legend_loc: {legend_loc}')
  # check input_dir exists
  if not exists(input_dir):
    print(f"input_dir: {input_dir} does not exist.")
    return -1
  # check the trace_data_latency.csv for each algorithm exists
  for algorithm in algorithms:
    if not exists(join(input_dir, f"trace_data_latency_{algorithm}.csv")):
      print(f"trace_data_latency_{algorithm}.csv does not exist.")
      return -1
  # read the trace_data_latency.csv for each algorithm into a dataframe, and then visualize the dataframe
  # use the first algorithm's tick, request_datetime
  df = pd.read_csv(join(input_dir, f"trace_data_latency_{algorithms[0]}.csv"))
  # https://sparkbyexamples.com/pandas/pandas-create-new-dataframe-by-selecting-specific-columns/
  combined_df = df[["tick", "request_datetime"]]
  # change column names, add algorithm name as suffix
  # https://datascienceparichay.com/article/pandas-rename-column-names/
  df = df.rename(columns=lambda x: f"{x}_{algorithms[0]}")
  combined_df = pd.concat(
    [combined_df, df[[*map(lambda x: f"{x}_{algorithms[0]}", [*metrics, *extra_metrics])]]], axis=1)
  for algorithm in algorithms[1:]:
    df = pd.read_csv(join(input_dir, f"trace_data_latency_{algorithm}.csv"))
    df = df.rename(columns=lambda x: f"{x}_{algorithm}")
    combined_df = pd.concat(
      [combined_df, df[[*map(lambda x: f"{x}_{algorithm}", [*metrics, *extra_metrics])]]], axis=1)
  visualization_from_dataframe(
    combined_df, algorithms, metrics, subplot, legend_loc)
  return 0


if __name__ == "__main__":
  # Make Python Fire not use a pager when it prints a help text
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
