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
from common.utility import get_latency_sync
from common.config_logging import init_logging

logger = init_logging(join(dirname(realpath(__file__)), "algorithm_ewma.log"))


class EACH_EWMA:
  def __init__(self, data: list[TraceData], file_metadata: dict[int: FileMetadata], N=6, n=3, k=2, ψ1=1, ψ2=1000, discount_factor=0.95, suffix=''):
    self.data = data
    self.file_metadata: dict[int: FileMetadata] = file_metadata
    self.N = N
    # only test the first 100 trace data if in debug environment
    self.ticks = len(data) if not debug else 10
    self.n = n
    self.k = k
    self.ψ1 = ψ1
    self.ψ2 = ψ2
    self.discount_factor = discount_factor
    self.suffix = suffix

  def processing(self):
    # initialization
    placement_policy_timed = np.zeros((self.ticks, self.N))
    latency_cloud_timed = np.zeros((self.ticks, self.N))
    current_ewma_latency = np.zeros((self.N,))
    C_N_n_count = len(list(itertools.combinations(range(self.N), self.n)))
    initial_optimized_placement = list(itertools.combinations(range(self.N), self.n))
    
    for tick, trace_data in enumerate(self.data):
      trace_data.tick = tick
      logger.info(f"[tick: {tick}]{'-'*20}")
      # make the first two placement_policy full use of the clould providers
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
        trace_data.latency_policy = current_ewma_latency.tolist()
        # sort ewma latency
        sorted_current_ewma_latency = np.argsort(current_ewma_latency)
        # Rank uˆi(t) in ascending order;
        # Select the top n arms to added into St for write operation
        # Select the top k arms based on placement to added into St for read operation
        if trace_data.file_read:
          # read operation
          placement = self.file_metadata[trace_data.file_id].placement
          # raise Exception when placement is empty
          if sum(placement) != self.n:
            raise Exception(f'Invalid placement {placement} for tick {tick}, trace_data: {trace_data}')
                
          placement_policy = np.zeros((self.N,), dtype=int)
          k = self.k
          for i, _ in enumerate(sorted_current_ewma_latency):
            if placement[i] == 1:
              placement_policy[i] = 1
              k -= 1
              if k == 0:
                break
          logger.info(
            f"current_ewma_latency: {current_ewma_latency}, sorted_current_ewma_latency: {sorted_current_ewma_latency}")
          logger.info(
            f"placement: {placement}, placement_policy: {placement_policy}")
        else:
          # write operation
          placement_policy = [
            1 if i in sorted_current_ewma_latency[:self.n] else 0 for i in range(self.N)]
          if self.file_metadata.get(trace_data.file_id) == None:
            self.file_metadata[trace_data.file_id] = FileMetadata(trace_data.offset, trace_data.file_size)
          file_metadata = self.file_metadata[trace_data.file_id]
          file_metadata.placement = placement_policy

      logger.info(f'placement policy: {placement_policy}')
      trace_data.request_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
      _, *latency_cloud = get_latency_sync(placement_policy, tick, self.N,
                                           self.k, cloud_providers, trace_data.file_size, trace_data.file_read)
      logger.info(f"latency_cloud: {latency_cloud}")
      # update the latency of trace_data
      trace_data.latency = max(latency_cloud)
      trace_data.latency_full = '   '.join(map(float_to_string, latency_cloud))
      trace_data.placement = '   '.join(map(str, self.file_metadata[trace_data.file_id].placement))
      trace_data.placement_policy = '_'.join(
        [str(i) for i, x in enumerate(placement_policy) if x == 1])
      placement_policy_timed[tick] = placement_policy
      latency_cloud_timed[tick] = latency_cloud

      choosed_cloud_ids = [i for i, x in enumerate(placement_policy) if x == 1]
      # update the ewma latency
      for cloud_id in choosed_cloud_ids:
        if current_ewma_latency[cloud_id] != 0:
          current_ewma_latency[cloud_id] = self.discount_factor * latency_cloud[cloud_id] + (
            1 - self.discount_factor) * current_ewma_latency[cloud_id]
        else:
          logger.warning(f'initial current_ewma_latency[{cloud_id}]')
          current_ewma_latency[cloud_id] = latency_cloud[cloud_id]

      logger.info(f'current_ewma_latency: {current_ewma_latency.tolist()}')
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
            
      # save the result interval
      if tick % 100 == 0:
          logger.info(f"tick: {tick}, save the result interval")
          self.save_result()

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
    with open(f'{results_dir}/trace_data_latency_ewma.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      header = ['tick', 'timestamp', 'file_id', 'file_size', 'file_read', 'placement', 'placement_policy',
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
  algorithm = EACH_EWMA(data, file_metadata,suffix=suffix)
  algorithm.processing()
  logger.info(f'processing finished')
  algorithm.save_result()
  logger.info(f'save_result finished')


if __name__ == "__main__":
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
