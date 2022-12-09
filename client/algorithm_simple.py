#!/usr/bin/env python3

import sys
from os.path import dirname, join, realpath, basename
from os import makedirs
import numpy as np
import fire
import csv
from datetime import datetime
from algorithm_common import *

sys.path.append(dirname(dirname(realpath(__file__))))
from common.config_logging import init_logging
from common.utility import get_latency_sync


logger = init_logging(
  join(dirname(realpath(__file__)), "algorithm_simple.log"))


class EACH_SIMPLE:
  def __init__(self, data: list[TraceData], file_metadata: dict[int: FileMetadata], N=6, n=3, k=2, ψ1=1, ψ2=1000,suffix=''):
    self.data = data
    self.file_metadata: dict[int: FileMetadata] = file_metadata
    self.N = N
    # only test the first 100 trace data if in debug environment
    self.ticks = len(data) if not debug else 10
    self.n = n
    self.k = k
    self.ψ1 = ψ1
    self.ψ2 = ψ2
    self.suffix = suffix

  def processing(self):
    # initialization
    placement_policy_timed = np.zeros((self.ticks, self.N))
    latency_cloud_timed = np.zeros((self.ticks, self.N))
    current_simple_latency = np.full((self.N,), np.inf)
    C_N_n_count = len(list(itertools.combinations(range(self.N), self.n)))
    initial_optimized_placement = list(itertools.combinations(range(self.N), self.n))
        
    for tick, trace_data in enumerate(self.data):
      trace_data.tick = tick
      logger.info(f"[tick: {tick}]{'-'*20}")
      # initial phase

      # # The first tick, the placement policy is selected randomly
      # if tick == 0:
      #   # write operation
      #   placement_policy = np.array([1, 1, 1, 0, 0, 0])
      # elif tick == 1:
      #   # write operation
      #   placement_policy = np.array([0, 0, 0, 1, 1, 1])
      if tick < C_N_n_count:
        # use full combinations matrix
        placement = [1 if i in initial_optimized_placement[tick] else 0 for i in range(self.N)]
        if self.file_metadata.get(trace_data.file_id) == None:
            self.file_metadata[trace_data.file_id] = FileMetadata(trace_data.offset, trace_data.file_size)
        file_metadata = self.file_metadata[trace_data.file_id]
        file_metadata.placement = placement
        placement_policy = placement
      else:
        trace_data.latency_policy = current_simple_latency.tolist()
        # sort simple latency
        sorted_current_simple_latency = np.argsort(current_simple_latency)
        # Rank uˆi(t) in ascending order;
        # Select the top n arms to added into St for write operation
        # Select the top k arms based on placement to added into St for read operation
        if trace_data.file_read:
          # read operation
          placement = self.file_metadata[trace_data.file_id].placement
          placement_policy = np.zeros((self.N,), dtype=int)
          k = self.k
          for i, _ in enumerate(sorted_current_simple_latency):
            if placement[i] == 1:
              placement_policy[i] = 1
              k -= 1
              if k == 0:
                break
          logger.info(
            f"current_simple_latency: {current_simple_latency}, sorted_current_simple_latency: {sorted_current_simple_latency}")
          logger.info(
            f"placement: {placement}, placement_policy: {placement_policy}")
        else:
          # write operation
          placement_policy = [
            1 if i in sorted_current_simple_latency[:self.n] else 0 for i in range(self.N)]

      logger.info(f'placement policy: {placement_policy}')
      # do request to get latency
      # make a request to the cloud and save the latency to the latency_cloud_timed
      # if the passed cloud_placements is like [0,0,1,0,1,0], then the returned latency is like [0,0,35.12,0,28.75,0]
      trace_data.request_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
      _, *latency_cloud = get_latency_sync(placement_policy, tick, self.N,
                                           self.k, cloud_providers, trace_data.file_size, trace_data.file_read)
      logger.info(f"latency_cloud: {latency_cloud}")
      # update the latency of trace_data
      trace_data.latency = max(latency_cloud)
      trace_data.latency_full = '   '.join(map(float_to_string, latency_cloud))
      trace_data.placement_policy = '_'.join(
        [str(i) for i, x in enumerate(placement_policy) if x == 1])
      placement_policy_timed[tick] = placement_policy
      latency_cloud_timed[tick] = latency_cloud

      # update the simple latency
      choosed_cloud_ids = [i for i, x in enumerate(placement_policy) if x == 1]
      # choosed_cloud_ids = np.where(placement_policy == 1)[0]
      for cloud_id in choosed_cloud_ids:
        current_simple_latency[cloud_id] = latency_cloud[cloud_id]

      if trace_data.file_read:
        post_reward = self.ψ1 * trace_data.latency + self.ψ2 * \
          sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 /
              1024 / self.k * outbound_cost[cloud_id], choosed_cloud_ids))
        trace_data.post_cost = sum(map(lambda cloud_id: trace_data.file_size /
                                   1024 / 1024 / 1024 / self.k * outbound_cost[cloud_id], choosed_cloud_ids))
      else:
        post_reward = self.ψ1 * trace_data.latency + self.ψ2 * \
          sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 /
              1024 / self.k * storage_cost[cloud_id], choosed_cloud_ids))
        trace_data.post_cost = sum(map(lambda cloud_id: trace_data.file_size /
                                   1024 / 1024 / 1024 / self.k * storage_cost[cloud_id], choosed_cloud_ids))
      trace_data.post_reward = post_reward
      logger.info(
        f"tick: {tick}, post_cost: {trace_data.post_cost}, post_reward: {post_reward}")

  def save_result(self):
    # create directory if not exists
    results_dir = join(dirname(realpath(__file__)), f'results_{self.suffix}')
    makedirs(results_dir, exist_ok=True)
    # update xxx_accumulated_average
    latency__accumulated_average = calculate_accumulated_average([trace_data.latency for trace_data in self.data if trace_data.tick != -1])
    post_reward_accumulated_average = calculate_accumulated_average([trace_data.post_reward for trace_data in self.data if trace_data.tick != -1])
    post_cost_accumulated_average = calculate_accumulated_average([trace_data.post_cost for trace_data in self.data if trace_data.tick != -1])
    post_cost_accumulation = calculate_accumulation([trace_data.post_cost for trace_data in self.data if trace_data.tick != -1])
    for index, trace_data in enumerate(filter(lambda trace_data: trace_data.tick != -1, self.data)):
        trace_data.post_reward_accumulated_average = post_reward_accumulated_average[index]
        trace_data.post_cost_accumulated_average = post_cost_accumulated_average[index]
        trace_data.post_cost_accumulation = post_cost_accumulation[index]
    # save the trace data with latency
    with open(f'{results_dir}/trace_data_latency_simple.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      header = ['timestamp', 'file_id', 'file_size', 'file_read', 'placement_policy',
                'latency', 'latency_full', 'post_reward', 'post_cost', 'request_datetime', 
                'post_reward_accumulated_average', 'post_cost_accumulated_average', 
                'post_cost_accumulation', 'latency_policy']
      writer.writerow(header)
      for trace_data in filter(lambda trace_data: trace_data.tick != -1, self.data):
        writer.writerow([getattr(trace_data, column) for column in header])


def main(input_file: str = join(dirname(realpath(__file__)), 'processed_test.txt')):
  # parsing the input file data
  test_data = TestData(input_file)
  data, file_metadata = test_data.load_data()
  logger.info(
    f'load_data data count: {len(data)}, file_metadata count: {len(file_metadata)}')
  file_metadata_list = list(file_metadata.items())
  logger.info(
    f'head of data: {data[:5]}, tail of data: {data[-5:]}, head of file_metadata: {file_metadata_list[:5]}, tail of file_metadata: {file_metadata_list[-5:]}')
  # run the algorithm
  suffix = basename(input_file).split('.')[0]
  algorithm = EACH_SIMPLE(data, file_metadata,suffix=suffix)
  algorithm.processing()
  logger.info(f'processing finished')
  algorithm.save_result()
  logger.info(f'save_result finished')


if __name__ == "__main__":
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
