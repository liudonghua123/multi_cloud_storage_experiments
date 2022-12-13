#!/usr/bin/env python3

import sys
from os.path import dirname, join, realpath, basename, splitext
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
  join(dirname(realpath(__file__)), "algorithm_random.log"))


class EACH_RANDOM:
  def __init__(self, data: list[TraceData], file_metadata: dict[int: FileMetadata], 
               N=N, n=n, k=k, ψ1=ψ1, ψ2=ψ2, suffix=''):
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
    current_random_latency = np.full((self.N,), np.inf)
    decision_metrics = np.zeros((self.N,))
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
        decision_metrics = list(range(self.N))
        random.shuffle(decision_metrics)
        # trace_data.latency_policy = decision_metrics.tolist()
        # sort random latency
        sorted_decision_metrics = np.argsort(decision_metrics)
        trace_data.decision_metrics = sorted_decision_metrics.tolist()
        # Rank uˆi(t) in ascending order;
        # Select the top n arms to added into St for write operation
        # Select the top k arms based on placement to added into St for read operation
        if trace_data.file_read:
          # read operation
          placement = self.file_metadata[trace_data.file_id].placement
          # raise Exception when placement is empty
          if sum(placement) != self.n:
            raise Exception(f'Invalid placement {placement} for tick {tick}, trace_data: {trace_data}')
          # if sorted_decision_metrics is like [4,2,3,5,1,0] and placement is like [0,1,1,0,0,1], then placement_policy is like [0,1,0,0,0,1]
          placement_policy = np.zeros((self.N,), dtype=int)
          k = self.k
          for i in sorted_decision_metrics:
            if placement[i] == 1:
              placement_policy[i] = 1
              k -= 1
              if k == 0:
                break
          logger.info(
            f"decision_metrics: {decision_metrics}, sorted_decision_metrics: {sorted_decision_metrics}")
          logger.info(
            f"placement: {placement}, placement_policy: {placement_policy}")
        else:
          # write operation
          placement_policy = [
            1 if i in sorted_decision_metrics[:self.n] else 0 for i in range(self.N)]
          if self.file_metadata.get(trace_data.file_id) == None:
            self.file_metadata[trace_data.file_id] = FileMetadata(trace_data.offset, trace_data.file_size)
          file_metadata = self.file_metadata[trace_data.file_id]
          file_metadata.placement = placement_policy

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
      # trace_data.placement = '   '.join(map(str, self.file_metadata[trace_data.file_id].placement))
      trace_data.placement = '_'.join(
        [str(i) for i, x in enumerate(self.file_metadata[trace_data.file_id].placement) if x == 1])
      trace_data.placement_policy = '_'.join(
        [str(i) for i, x in enumerate(placement_policy) if x == 1])
      placement_policy_timed[tick] = placement_policy
      latency_cloud_timed[tick] = latency_cloud

      # update the random latency
      choosed_cloud_ids = [i for i, x in enumerate(placement_policy) if x == 1]
      
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
    with open(f'{results_dir}/trace_data_latency_random.csv', 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      header = ['tick', 'timestamp', 'file_id', 'file_size', 'file_read', 'placement', 'placement_policy',
                'latency', 'latency_full', 'post_reward', 'post_cost', 'request_datetime', 
                'post_reward_accumulated_average', 'post_cost_accumulated_average', 
                'post_cost_accumulation', 'latency_policy', 'decision_metrics']
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
  suffix = splitext(basename(input_file))[0]
  algorithm = EACH_RANDOM(data, file_metadata,suffix=suffix)
  algorithm.processing()
  logger.info(f'processing finished')
  algorithm.save_result()
  logger.info(f'save_result finished')


if __name__ == "__main__":
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
