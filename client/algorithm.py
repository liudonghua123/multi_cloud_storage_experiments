#!/usr/bin/env python3

import sys
import time
from os.path import dirname, join, realpath
import itertools
import math
import numpy as np
import asyncio
import fire
import csv
from datetime import datetime
from algorithm_common import *

sys.path.append(dirname(dirname(realpath(__file__))))

from common.utility import get_latency, get_latency_sync
from common.config_logging import init_logging

logger = init_logging(join(dirname(realpath(__file__)), "client.log"))
    
# 
# Use the following variable names to store the data
# N: the number of clouds
# n: write_pieces
# k: read_pieces, n > k
# w0: default_window_size
# δ: delta
# b: threshold
# λ: lambda
# St: placement_policy_timed is (T,N) matrix
# τi: changed_ticks is (N,) vector
# wi: window_sizes is (N,) matrix
# ξ: xi
# ψ: psi
# Wit: window_sizes_timed is (T,N) matrix
# lit: window_sizes_timed is (T,N) matrix
# T: ticks, the number of ticks in the simulation
#
class AW_CUCB:
    def __init__(self, data: list[TraceData], file_metadata: dict[int: FileMetadata],default_window_size=50, N=6, n=3, k=2, ψ1=1, ψ2=1000, ξ=1, b_increase=0.4, b_decrease=0.4, δ=0.5, optimize_initial_exploration=True, LB=None):
        self.data = data
        self.default_window_size = default_window_size
        self.file_metadata: dict[int: FileMetadata] = file_metadata
        self.N = N
        # only test the first 100 trace data if in debug environment
        self.ticks=len(data) if not debug else 10
        self.n = n
        self.k = k
        self.ψ1 = ψ1
        self.ψ2 = ψ2
        self.ξ = ξ
        self.b_increase = b_increase
        self.b_decrease = b_decrease
        self.δ = δ
        self.optimize_initial_exploration = optimize_initial_exploration
        self.LB = LB
        self.migration_records: list[MigrationRecord] = []
        self.change_point_records: list[ChangePointRecord] = []
        self.last_change_tick: list[int] = [0] * self.N
        
        
    def processing(self):
        # initialization
        τ = np.full((self.N,),1)
        window_sizes = np.full((self.N,),self.default_window_size)
        placement_policy_timed = np.zeros((self.ticks, self.N))
        latency_cloud_timed = np.zeros((self.ticks, self.N))
        U = np.zeros((self.ticks + 1, self.N))
        L = np.zeros((self.ticks + 1, self.N))
        U_min = np.zeros((self.ticks + 1, self.N))
        L_max = np.zeros((self.ticks + 1, self.N))
        C_N_n_count = len(list(itertools.combinations(range(self.N), self.n)))
        Tiwi = np.zeros((self.N,))
        liwi = np.zeros((self.N,))
        LB = np.zeros((self.N,))
        eit = np.zeros((self.N,))
        u_hat_it = np.zeros((self.N,))
        
        for tick, trace_data in enumerate(self.data):
            
            # exploration phase
            if tick < C_N_n_count:
                # write operation
                placement_policy = self.file_metadata[trace_data.file_id].placement
            # ulization phase
            else:
                # sort uit
                sorted_u_hat_it = np.argsort(u_hat_it)
                # Rank uˆi(t) in ascending order; 
                # Select the top n arms to added into St for write operation
                # Select the top k arms based on placement to added into St for read operation
                if trace_data.file_read:
                    # read operation
                    placement = self.file_metadata[trace_data.file_id].placement
                    # raise Exception when placement is empty
                    if sum(placement) != self.n:
                        raise Exception(f'Invalid placement {placement} for tick {tick}, trace_data: {trace_data}')
                    # if placement is [0,1,1,0,1,0] and sorted_u_hat_it is [0,1,2,3,4,5], then the top k arms are [1,2], placement_policy is [0,1,1,0,0,0]
                    placement_policy = np.zeros((self.N,), dtype = int)
                    k = self.k
                    for i in sorted_u_hat_it:
                        if placement[i] == 1:
                            placement_policy[i] = 1
                            k -= 1
                            if k == 0:
                                break
                    logger.info(f'placement: {placement}, sorted_u_hat_it: {sorted_u_hat_it}, placement_policy: {placement_policy}')
                else:
                    # write operation
                    placement_policy = [1 if i in sorted_u_hat_it[:self.n] else 0 for i in range(self.N)]
                    
            # do request to get latency
            choosed_cloud_ids = [i for i, x in enumerate(placement_policy) if x == 1]
            # make a request to the cloud and save the latency to the latency_cloud_timed
            # if the passed cloud_placements is like [0,0,1,0,1,0], then the returned latency is like [0,0,35.12,0,28.75,0]
            # Use coroutine to make request
            # _, *latency_cloud = asyncio.run(get_latency(placement_policy, tick, self.N, self.k, cloud_providers, trace_data.file_size, trace_data.file_read))
            # Use thread to make request
            _, *latency_cloud = get_latency_sync(placement_policy, tick, self.N, self.k, cloud_providers, trace_data.file_size, trace_data.file_read)
            logger.info(f"tick: {tick}, latency_cloud: {latency_cloud}")
            # update the latency of trace_data
            trace_data.latency = max(latency_cloud)
            trace_data.latency_full = '   '.join(map(float_to_string, latency_cloud))
            trace_data.placement_policy = '   '.join(map(str, placement_policy))
            placement_policy_timed[tick] = placement_policy   
            latency_cloud_timed[tick] = latency_cloud
            # update statistics 17
            # Update statistics in time-window Wi(t) according to (17);
            # choosed_cloud_ids = np.where(placement_policy == 1)[0]
            for cloud_id in choosed_cloud_ids:
                start_tick = max(0, tick - window_sizes[cloud_id])
                Tiwi[cloud_id] = np.sum(placement_policy_timed[start_tick: tick + 1, cloud_id], axis=0)
                latency_cloud_previous = latency_cloud_timed[start_tick: tick + 1, cloud_id]
                liwi[cloud_id] = 1 / Tiwi[cloud_id] * np.sum(latency_cloud_previous, axis=0)
                LB[cloud_id] = max_except_zero(latency_cloud_previous) - min_except_zero(latency_cloud_previous) if self.LB == None else self.LB
                eit[cloud_id] = LB[cloud_id] * math.sqrt(self.ξ * math.log(window_sizes[cloud_id], 10) / Tiwi[cloud_id])
                
                # Estimate/Update the utility bound for each i ∈ [N], TODO: update uit # latency / data_size
                # np_array[:]=list() will not change the datetype of np_array, while np_array=list() will change.
                # however, if some operands are np_array, then np_array=a*b+c will keep the datetype of np_array
                if trace_data.file_read:
                    u_hat_it[cloud_id] = self.ψ1 * liwi[cloud_id] + self.ψ2 * (trace_data.file_size / 1024 / 1024 / 1024 / self.k * outbound_cost[cloud_id]) - eit[cloud_id]
                else:
                    u_hat_it[cloud_id] = self.ψ1 * liwi[cloud_id] + self.ψ2 * (trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[cloud_id]) - eit[cloud_id]
            
            trace_data.LB = '   '.join(map(float_to_string, LB))
            trace_data.eit = '   '.join(map(float_to_string, eit))
            trace_data.u_hat_it = '   '.join(map(float_to_string, u_hat_it))
            logger.info(f"tick: {tick}, u_hat_it: {u_hat_it}")
            if trace_data.file_read:
                post_reward = self.ψ1 * trace_data.latency + self.ψ2 * sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 / 1024 / self.k * outbound_cost[cloud_id], choosed_cloud_ids))
                trace_data.post_reward = f'{post_reward}={self.ψ1}*{trace_data.latency}+{self.ψ2}*{sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 / 1024 / self.k * outbound_cost[cloud_id], choosed_cloud_ids))}'
                trace_data.post_cost = sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 / 1024 / self.k * outbound_cost[cloud_id], choosed_cloud_ids))
            else: 
                post_reward = self.ψ1 * trace_data.latency + self.ψ2 * sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[cloud_id], choosed_cloud_ids))
                trace_data.post_reward = f'{post_reward}={self.ψ1}*{trace_data.latency}+{self.ψ2}*{sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[cloud_id], choosed_cloud_ids))}'
                trace_data.post_cost = sum(map(lambda cloud_id: trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[cloud_id], choosed_cloud_ids))
            logger.info(f"tick: {tick}, post_reward: {post_reward}")
            
            # check whether FM_PHT
            changed_ticks = self.FM_PHT(U, L, U_min, L_max, tick, latency_cloud_timed)
            logger.info(f"tick: {tick}, changed_ticks: {changed_ticks}")
            # St' = file_metadata[file_id].placement, donote as previous_placement_policy
            St_hat = self.file_metadata[trace_data.file_id].placement
            # St = select the top n from N in uit, donote as current_placement_policy
            St = [1 if i in np.argsort(u_hat_it)[:self.n] else 0 for i in range(self.N)]
            trace_data.migration_targets = '   '.join(map(str, St))
            if any(changed_ticks):
                # convert changed_ticks from ChangePoint to int
                logger.info(f'before convert, changed_ticks: {changed_ticks}')
                changed_ticks = list(map(lambda x: x.tick if x != None else 0, changed_ticks))
                logger.info(f'after convert, changed_ticks: {changed_ticks}')
                # save the change point
                self.change_point_records.append(ChangePointRecord(tick, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '   '.join(map(float_to_string, changed_ticks)), '   '.join([str(i) if v != 0 else '0' for i, v in enumerate(changed_ticks)])))
                # update τ from FM_PHT result
                τ = np.array(changed_ticks)
                logger.info(f'tick: {tick}, τ: {τ}')
                # if read operation
                if trace_data.file_read:
                    # LDM(St', St), ST: current placement_policy, ST': the previous placement_policy
                    self.LDM(tick, trace_data, St_hat, St)
            # update window size according to τ
            logger.info(f'tick: {tick}, before update window_sizes: {window_sizes}, τ: {τ}')
            window_sizes = np.minimum(self.default_window_size, tick + 1 - τ + 1)
            logger.info(f'tick: {tick}, after update window_sizes: {window_sizes}, τ: {τ}')
            print(f"tick: {tick}, window_sizes: {window_sizes}")
            
            # save the result interval
            if tick % 100 == 0:
                logger.info(f"tick: {tick}, save the result interval")
                self.save_result()
                
    def LDM(self, tick, trace_data: TraceData, previous_placement_policy, current_placement_policy):
        # convert the placement_policy to the selected cloud providers
        current_placement_policy_indices = set(np.where(current_placement_policy == 1)[0].tolist())
        previous_placement_policy_indices = set(np.where(previous_placement_policy == 1)[0].tolist())
        prepare_migrate_cloud_ids = previous_placement_policy_indices - current_placement_policy_indices
        destination_migrate_cloud_ids = current_placement_policy_indices - previous_placement_policy_indices
        logger.info(f"current_placement_policy: {current_placement_policy}, current_placement_policy_indices: {current_placement_policy_indices}, previous_placement_policy: {previous_placement_policy}, previous_placement_policy_indices: {previous_placement_policy_indices}, prepare_migrate_cloud_ids: {prepare_migrate_cloud_ids}, destination_migrate_cloud_ids: {destination_migrate_cloud_ids}")
        # initial migration gains to 0
        migration_gains = 0
        if len(prepare_migrate_cloud_ids) > 0:
            # calculate migration gains
            migration_gains = sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * (storage_cost[i] - outbound_cost[i]), prepare_migrate_cloud_ids)) - sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[i], destination_migrate_cloud_ids))
            # calculate migration cost
            migration_cost = sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * outbound_cost[i], prepare_migrate_cloud_ids)) + sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[i], destination_migrate_cloud_ids))
            trace_data.migration_gains = migration_gains
            trace_data.migration_cost = migration_cost
        if migration_gains > 0:
            logger.info(f'perform migration from {prepare_migrate_cloud_ids} to {destination_migrate_cloud_ids} at tick {tick}')
            # migrate the data from prepare_migrate_cloud_ids (read) to destination_migrate_cloud_ids (write)
            # process the migration, record processed latency async
            start_time = time.time()
            _, *latency_cloud_read = asyncio.run(get_latency([1 if i in prepare_migrate_cloud_ids else 0 for i in range(self.N)], tick, self.N, self.k, cloud_providers, trace_data.file_size, True))
            _, *latency_cloud_write = asyncio.run(get_latency([1 if i in destination_migrate_cloud_ids else 0 for i in range(self.N)], tick, self.N, self.k, cloud_providers, trace_data.file_size, False))
            latency = int((time.time() - start_time) * 1000)
            logger.info(f"latency_cloud_read: {latency_cloud_read}, latency_cloud_write: {latency_cloud_write}, total latency: {latency}")
            # update the file_metadata
            logger.info(f'update the file_metadata at tick {tick}')
            self.file_metadata[trace_data.file_id].placement = current_placement_policy
            self.migration_records.append(MigrationRecord(trace_data.file_id, tick, datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f'), latency, migration_cost, migration_gains))
 
    def test_FM_PHT(self, test_csv_file: str):
        # parse the test csv file
        latency_cloud_timed = []
        with open(test_csv_file, 'r') as f:
            reader = csv.reader(f)
            # skip the header
            next(reader)
            for row in reader:
                latency_cloud_timed.append(list(map(float, row)))
        # invoke FM_PHT
        ticks = len(latency_cloud_timed)
        self.N = 4
        latency_cloud_timed = np.array(latency_cloud_timed)
        U = np.zeros((ticks + 1, self.N))
        L = np.zeros((ticks + 1, self.N))
        U_min = np.zeros((ticks + 1, self.N))
        L_max = np.zeros((ticks + 1, self.N))
        detected_ticks = []
        # self.δ = 0.9    
        for tick in range(ticks):
            changed_ticks = self.FM_PHT(U, L, U_min, L_max, tick, latency_cloud_timed)
            if any(changed_ticks):
                detected_ticks.append(tick)
                logger.info(f'tick: {tick}, changed_ticks: {changed_ticks}')
        logger.info(f'self.δ: {self.δ}, detected_ticks count: {len(detected_ticks)}, {detected_ticks}')
        self.save_matrix_as_csv(U_min, 'U_min.csv')
        self.save_matrix_as_csv(L_max, 'L_max.csv')
        self.save_matrix_as_csv(U, 'U.csv')
        self.save_matrix_as_csv(L, 'L.csv')
        
    # save the matrix as csv file
    def save_matrix_as_csv(self, matrix, file_name):
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(matrix)
                 
    def FM_PHT(self, U, L, U_min, L_max, tick, latency_cloud_timed):
        tick += 1
        # logger.info(f'U_min:tick,  {U_min}tick, , L_max: {L_max}')
        # U[0]={0}, L[0]={0}, the tick start from 1
        
        def find_exists_value_backward(array, row, column):
            while row >= 0:
                if array[row][column] != 0:
                    return array[row][column]
                row -= 1
            return 0
        
        changed_ticks: list[ChangePoint] = []
        for cloud_id in range(self.N):
            # skip the cloud_id that is not used
            if latency_cloud_timed[tick - 1, cloud_id] == 0:
                changed_ticks.append(None)
                continue
            latency_cloud = latency_cloud_timed[self.last_change_tick[cloud_id]:tick, cloud_id]
            latency_cloud_exist = np.delete(latency_cloud, np.where(latency_cloud == 0))
            
            U[tick][cloud_id] = (tick - 1) / tick * find_exists_value_backward(U, tick-1, cloud_id) + (latency_cloud_exist[-1] - np.average(latency_cloud_exist) - self.δ)
            
            L[tick][cloud_id] = (tick - 1) / tick * find_exists_value_backward(L, tick-1, cloud_id) + (latency_cloud_exist[-1] - np.average(latency_cloud_exist) + self.δ)
            
            U_min[tick-1, cloud_id] = min_except_zero(U[:tick + 1, cloud_id])
            L_max[tick-1, cloud_id] = max_except_zero(L[:tick + 1, cloud_id])
            changed_tick = None
            if U[tick, cloud_id] - U_min[tick-1, cloud_id] >= self.b_increase:
                changed_tick = ChangePoint(argmin_except_zero(U[:tick, cloud_id]) , ChangePoint.INCREASE)
            if L_max[tick-1, cloud_id] - L[tick, cloud_id] >= self.b_decrease:
                if changed_tick != None:
                    #save the U_min and L_max, U and L matrix        
                    self.save_matrix_as_csv(U_min, 'U_min.csv')
                    self.save_matrix_as_csv(L_max, 'L_max.csv')
                    self.save_matrix_as_csv(U, 'U.csv')
                    self.save_matrix_as_csv(L, 'L.csv')
                    logger.info(f'latency_cloud_timed[self.last_change_tick[cloud_id]:tick]: \n{latency_cloud_timed[self.last_change_tick[cloud_id]:tick]}')
                    logger.info(f'\nU[:tick + 1]: \n{U[:tick + 1]}, \nL[:tick + 1]: \n{L[:tick + 1]}, \nU_min[:tick]: \n{U_min[:tick]}, \nL_max[:tick]: \n{L_max[:tick]}')
                    logger.info(f'\nU[tick, cloud_id] - U_min[tick-1, cloud_id]: {U[tick, cloud_id] - U_min[tick-1, cloud_id]}\nL_max[tick-1, cloud_id] - L[tick, cloud_id]: {L_max[tick-1, cloud_id] - L[tick, cloud_id]}')
                    raise RuntimeError(f'tick: {tick}, could_id: {cloud_id}, U and L both changed, this should not happen')
                changed_tick = ChangePoint(argmax_except_zero(L[:tick, cloud_id]), ChangePoint.DECREASE)
            # if changed_tick != None:
            #     logger.info(f'tick: {tick - 1}, cloud_id: {cloud_id}, changed_tick: {changed_tick}')
            changed_ticks.append(changed_tick)
        
        # reset FM-PHT
        if any(changed_ticks):
            logger.info(f'tick: {tick - 1}, current latency_cloud: {latency_cloud_timed[tick - 1]}, latency_cloud_timed[tick - 5: tick + 5]: \n{latency_cloud_timed[tick - 5: tick + 5]}')
            
            for index, changed_tick in enumerate(changed_ticks):
                if changed_tick == None:
                    continue
                
                self.last_change_tick[index] = changed_tick.tick
                if changed_tick.type == ChangePoint.INCREASE:
                    U[:changed_tick.tick + 1, index] = 0
                    U_min[:changed_tick.tick, index] = 0
                    U_min[changed_tick.tick, index] = min_except_zero(U[changed_tick.tick + 1: tick + 1, index])
                    # latency_cloud_timed[:changed_tick.tick, index] = 0
                elif changed_tick.type == ChangePoint.DECREASE:
                    L[:changed_tick.tick + 1, index] = 0
                    L_max[:changed_tick.tick, index] = 0
                    L_max[changed_tick.tick, index] = max_except_zero(L[changed_tick.tick + 1: tick + 1, index])
                    # latency_cloud_timed[:changed_tick.tick, index] = 0
            
            # save the U_min and L_max, U and L matrix        
            # self.save_matrix_as_csv(U_min, 'U_min.csv')
            # self.save_matrix_as_csv(L_max, 'L_max.csv')
            # self.save_matrix_as_csv(U, 'U.csv')
            # self.save_matrix_as_csv(L, 'L.csv')
                    
        return changed_ticks
        
        
    def save_result(self):
        # save the migration records
        with open('results/migration_records.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['file_id', 'tick', 'start_time', 'latency', 'migration_gains', 'migration_cost']
            writer.writerow(header)
            for migration_record in self.migration_records:
                writer.writerow([getattr(migration_record, column) for column in header])
        # save the trace data with latency
        with open('results/trace_data_latency.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['timestamp', 'file_id', 'file_size', 'file_read', 'latency', 'latency_full', 'placement_policy', 'migration_targets' ,'post_reward', 'post_cost', 'LB', 'eit', 'u_hat_it', 'migration_gains', 'migration_cost']
            writer.writerow(header)
            for trace_data in self.data:
                writer.writerow([getattr(trace_data, column) for column in header])
        # save the change points
        with open('results/change_points.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['tick', 'datetime', 'change_point_tick', 'change_cloud_providers']
            writer.writerow(header)
            for change_point_record in self.change_point_records:
                writer.writerow([getattr(change_point_record, column) for column in header])
    
    
def main(input_file: str = join(dirname(realpath(__file__)), 'processed_test.txt')):
    # parsing the input file data
    test_data = TestData(input_file)
    data, file_metadata = test_data.load_data()
    logger.info(f'load_data data count: {len(data)}, file_metadata count: {len(file_metadata)}')
    file_metadata_list = list(file_metadata.items())
    logger.info(f'head of data: {data[:5]}, tail of data: {data[-5:]}, head of file_metadata: {file_metadata_list[:5]}, tail of file_metadata: {file_metadata_list[-5:]}')
    # run the algorithm
    start_time = time.time()
    algorithm = AW_CUCB(data, file_metadata)
    algorithm.processing()
    logger.info(f'processing finished')
    algorithm.save_result()
    logger.info(f'save_result finished')
    logger.info(f'total time: {time.time() - start_time}')

def test(test_csv_file: str = 'network_test_results/network_test_aggregated__mean.csv'):
    algorithm = AW_CUCB([], [])
    algorithm.test_FM_PHT(test_csv_file)
    
if __name__ == "__main__":
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
