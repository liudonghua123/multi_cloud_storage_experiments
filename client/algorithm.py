#!/usr/bin/env python3

from dataclasses import dataclass
import sys
import time
import os
from os.path import dirname, join, realpath
import itertools
import math
import numpy as np
# ndarray for type hints
from numpy import ndarray
from numpy import nan
import yaml
import random
import snoop
import asyncio
import requests
from requests.exceptions import Timeout, ConnectionError
import fire
import csv
from typing import TypedDict
from datetime import datetime

# serialize and deserialize using jsonpickle or pickle
# jsonpickle is better for human readable
# pickle is better for performance and smaller size
import jsonpickle # jsonpickle.dumps / jsonpickle.loads
import pickle # pickle.dumps / pickle.loads

# In order to run this script directly, you need to add the parent directory to the sys.path
# Or you need to run this script in the parent directory using the command: python -m client.algorithm
sys.path.append(dirname(dirname(realpath(__file__))))

from common.utility import get_latency, spinner_context, get_config
from common.config_logging import init_logging

logger = init_logging(join(dirname(realpath(__file__)), "client.log"))

config = get_config(dirname(realpath(__file__)))
logger.info(f"load config: {config}")

storage_cost : list[float] = config['storage_cost']
outbound_cost : list[float] = config['outbound_cost']
read_cost : list[float] = config['read_cost']
write_cost : list[float] = config['write_cost']
cloud_providers : list[str] = config['cloud_providers']
debug : bool = config['debug']


@dataclass
class FileMetadata:
    offset: int
    size: int
    placement: list[int]
    
@dataclass
class MigrationRecord:
    file_id: int
    tick: int
    start_time: str
    latency: int
    migration_gains: float
    migration_cost: float
    
@dataclass
class ChangePointRecord:
    tick: int
    datetime: str
    change_point_tick: list[int]
    change_cloud_providers: list[int]
    
    
@dataclass
class TraceData:
    # timestamp in seconds
    timestamp: int
    file_id: int
    offset: int
    file_size: int
    file_read: bool = True
    datetime_offset: int = 0
    latency: int = -1
    placement_policy: list[int] = None
    
def get_file_line_count(file_path):
    with open(file_path, 'rb') as fp:
        def _read(reader):
            buffer_size = 1024 * 1024
            b = reader(buffer_size)
            while b:
                yield b
                b = reader(buffer_size)
        content_generator = _read(fp.raw.read)
        count = sum(buffer.count(b'\n') for buffer in content_generator)
        return count

class TestData:
    '''
    Load the test data which is preprocess by the preprocess.py
    The data structure is like this:
    timestamp,datetime,hostname,disk_number,type,offset,size,response_time
    128166477394345573,2010-08-13 09:59:33.943456,hm,1,Read,383496192,32768,113736
    128166483087163644,2010-08-13 10:00:30.871636,hm,1,Read,2822144,65536,71730
    128166620794097839,2010-08-13 10:23:27.940978,hm,1,Read,3221266432,4096,121008
    
    Convert the data to a list of dictionary, each dictionary is a row of the csv file, only keep the useful columns(timeStamp, type, file_id as offset, size) and DateTimeOffset based on the first timestamp in one second period.
    
    TODO: the current implementation store the data in memory in order to use it efficiently, but it may cause memory issue if the data is too large or could not make it persistence. 
    We may need to use a database or key-value store like redis to store the data.
    '''
    
    default_data_file: str = 'data.bin'
    default_file_metadata_file: str = 'file_metadata.bin'
    
    def __init__(self, file_path: str, N=6, n=3, k=2, size_enlarge=100):
        '''
        file_path: the path of the test data file
        '''
        self.file_path = file_path
        self.N = N
        self.n = n
        self.k = k
        self.size_enlarge = size_enlarge
        self.data = []
        self.file_metadata: dict[int: FileMetadata] = {}
    
    def load_data(self):
        if os.path.exists(self.default_data_file) and os.path.exists(self.default_file_metadata_file):
            logger.info(f'using the serilized data')
            return TestData._load_data_via_deserialize()
        logger.info(f'parse {self.file_path} data')
        data, file_metadata = self._parse_data()
        logger.info(f'serialize the parsed to {self.default_data_file} and {self.default_file_metadata_file}')
        TestData._save_data_via_serialize(data, file_metadata)
        return data, file_metadata
    
    @staticmethod
    def _load_data_via_deserialize(data_file: str = default_data_file, file_metadata_file: str = default_file_metadata_file):
        with open(data_file, 'rb') as data_file_fd, open(file_metadata_file, 'rb') as file_metadata_file_fd:
            data = pickle.load(data_file_fd)
            file_metadata = pickle.load(file_metadata_file_fd)
        return data, file_metadata
    
    @staticmethod
    def _save_data_via_serialize(data, file_metadata, data_file: str = default_data_file, file_metadata_file: str = default_file_metadata_file):
        with open(data_file, 'wb') as data_file_fd, open(file_metadata_file, 'wb') as file_metadata_file_fd:
            pickle.dump(data, data_file_fd)
            pickle.dump(file_metadata, file_metadata_file_fd)
    
    def _parse_data(self):
        # get the number of lines in the file
        self.file_input_lines = get_file_line_count(self.file_path)
        # read one line to get the first timestamp
        with open(self.file_path) as fin:
            first_timestamp, *_ = fin.readline().split(',')
            self.initial_timestamp = int(first_timestamp[:10])
        logger.info(f'file_input_lines: {self.file_input_lines}, initial timestamp: {self.initial_timestamp}')
        with open(self.file_path) as fin, spinner_context('Processing the data ...') as spinner:
            # update the spinner text to show the progress in 00.01% minimum
            update_tick = int(self.file_input_lines / 100)
            write_processed_count = 0
            for index, line in enumerate(fin):
                timestamp,_,_,_,operation_type,offset,size,_ = line.split(',')
                timestamp = int(timestamp[:10])
                # use offset and size combination as file_id, use int type
                file_id = int(f'{offset}{size}')
                offset = int(offset)
                # enlarge the size by 100 times
                size = int(size) * self.size_enlarge
                read = operation_type == 'Read'
                initial_optimized_placement = list(itertools.combinations(range(self.N), self.n))
                # if the file_id is not in the file_metadata, add it to the file_metadata
                if self.file_metadata.get(file_id) is None:
                    self.file_metadata[file_id] = FileMetadata(offset, size, None)
                # update placement of file_metadata, if the original placement is None and the new placement is not None
                file_metadata = self.file_metadata[file_id]
                if file_metadata.placement is None:
                    # 1. The placement for read will be randomly selected
                    # 2. The placement for write: 
                    # 2.1 Overwrite the first C_N_n placement for write operation, using the initial_optimized_placement
                    # 2.2 The rest placement for write will be not set empty [], maybe the last operation for such file is write, look 383848448,8192, select by algotithm
                    if write_processed_count < len(initial_optimized_placement):
                        read = False
                        placement = [1 if i in initial_optimized_placement[write_processed_count] else 0 for i in range(self.N)]
                        write_processed_count += 1
                    elif read:
                        placement = random.choice(initial_optimized_placement)
                        placement = [1 if i in placement else 0 for i in range(self.N)]
                        if (read or index < len(initial_optimized_placement)) and (sum(placement) != self.n):
                            raise Exception(f'Invalid placement {placement} for index, line: {index}, {line}, read: {read}, n: {self.n}, k: {self.k}, N: {self.N}')
                    if placement is not None:
                        file_metadata.placement = placement
                        
                self.data.append(TraceData(timestamp, file_id, offset, size, read, timestamp - self.initial_timestamp))
                if index % update_tick == 0:
                    spinner.text = f'Processing {index / self.file_input_lines * 100:.2f}%'
        return self.data, self.file_metadata
    
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
# Wit: windows_sizes_timed is (T,N) matrix
# lit: windows_sizes_timed is (T,N) matrix
# T: ticks, the number of ticks in the simulation
#
class AW_CUCB:
    def __init__(self, data: list[TraceData], file_metadata: dict[int: FileMetadata],default_window_size=50, N=6, n=3, k=2, ψ1=0.5, ψ2=0.5, ξ=0.5, b_increase=0.4, b_decrease=0.3, δ=0.05, optimize_initial_exploration=True, LB=None):
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
        
        
    def processing(self):
        # initialization
        τ = np.full((self.N,),1)
        window_sizes = np.full((self.N,),self.default_window_size)
        placement_policy_timed = np.zeros((self.ticks, self.N))
        windows_sizes_timed = np.zeros((self.ticks, self.N))
        latency_cloud_timed = np.zeros((self.ticks, self.N))
        U = np.zeros((self.ticks + 1, self.N))
        L = np.zeros((self.ticks + 1, self.N))
        C_N_n_count = len(list(itertools.combinations(range(self.N), self.n)))
        Tiwi = np.zeros((self.N,))
        liwi = np.zeros((self.N,))
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
            # make a request to the cloud and save the latency to the latency_cloud_timed
            # if the passed cloud_placements is like [0,0,1,0,1,0], then the returned latency is like [0,0,35.12,0,28.75,0]
            _, *latency_cloud = asyncio.run(get_latency(placement_policy, tick, self.N, self.k, cloud_providers, trace_data.file_size, trace_data.file_read))
            logger.info(f"tick: {tick}, latency_cloud: {latency_cloud}")
            # update the latency of trace_data
            trace_data.latency = max(latency_cloud)
            trace_data.placement_policy = '_'.join([str(i) for i, x in enumerate(placement_policy) if x == 1])
            placement_policy_timed[tick] = placement_policy   
            latency_cloud_timed[tick] = latency_cloud
            # update statistics 17
            # Update statistics in time-window Wi(t) according to (17);
            choosed_clould_ids = [i for i, x in enumerate(placement_policy) if x == 1]
            # choosed_clould_ids = np.where(placement_policy == 1)[0]
            for clould_id in choosed_clould_ids:
                Tiwi[clould_id] = np.sum(placement_policy_timed[:tick + 1,clould_id], axis=0)
                latency_of_cloud_previous_ticks = latency_cloud_timed[:tick + 1, clould_id]
                liwi[clould_id] = 1 / Tiwi[clould_id] * np.sum(latency_of_cloud_previous_ticks, axis=0)
                LB = latency_of_cloud_previous_ticks.max() - np.delete(latency_of_cloud_previous_ticks, np.where(latency_of_cloud_previous_ticks == 0)).min() if self.LB == None else self.LB
                eit[clould_id] = LB * math.sqrt(self.ξ * math.log(window_sizes[clould_id], 10) / Tiwi[clould_id])
                
                # Estimate/Update the utility bound for each i ∈ [N], TODO: update uit # latency / data_size
                # np_array[:]=list() will not change the datetype of np_array, while np_array=list() will change.
                # however, if some operands are np_array, then np_array=a*b+c will keep the datetype of np_array
                if trace_data.file_read:
                    u_hat_it[clould_id] = self.ψ1 * liwi[clould_id] + self.ψ2 * (trace_data.file_size / 1024 / 1024 / 1024 / self.n * outbound_cost[clould_id]) - eit[clould_id]
                else:
                    u_hat_it[clould_id] = self.ψ1 * liwi[clould_id] + self.ψ2 * (trace_data.file_size / 1024 / 1024 / 1024 / self.n * storage_cost[clould_id]) - eit[clould_id]
            logger.info(f"tick: {tick}, u_hat_it: {u_hat_it}")
            
            # check whether FM_PHT
            changed, changed_ticks = self.FM_PHT(U,L,tick,latency_cloud_timed)
            logger.info(f"tick: {tick}, changed: {changed}, changed_ticks: {changed_ticks}")
            if any(changed):
                # save the change point
                self.change_point_records.append(ChangePointRecord(tick, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '_'.join([str(i) for i in changed_ticks if i != 0]), '_'.join([str(i) for i, x in enumerate(changed) if x == 1])))
                # update τ from FM_PHT result
                τ = changed_ticks
                logger.info(f'tick: {tick}, τ: {τ}')
                # TODO: reset FM-PHT
                # if read operation
                if trace_data.file_read:
                    # St' = file_metadata[file_id].placement, donote as previous_placement_policy
                    St_hat = self.file_metadata[trace_data.file_id].placement
                    # St = select the top n from N in uit, donote as current_placement_policy
                    St = [1 if i in np.argsort(u_hat_it)[:self.n] else 0 for i in range(self.N)]
                    # LDM(St', St), ST: current placement_policy, ST': the previous placement_policy
                    self.LDM(tick, trace_data, St_hat, St)
            # update window size according to τ
            logger.info(f'tick: {tick}, before update window_sizes: {window_sizes}, τ: {τ}')
            window_sizes = np.minimum(self.default_window_size, tick + 1 - τ + 1)
            logger.info(f'tick: {tick}, after update window_sizes: {window_sizes}, τ: {τ}')
            print(f"tick: {tick}, window_sizes: {window_sizes}")
                
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
            migration_gains = sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * (storage_cost[i] - outbound_cost[i]) - read_cost[i], prepare_migrate_cloud_ids)) - sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[i] + write_cost[i], destination_migrate_cloud_ids))
            # calculate migration cost
            migration_cost = sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * outbound_cost[i] + read_cost[i], prepare_migrate_cloud_ids)) + sum(map(lambda i: trace_data.file_size / 1024 / 1024 / 1024 / self.k * storage_cost[i] + write_cost[i], destination_migrate_cloud_ids))
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
            
    def FM_PHT(self, U, L, tick, latency_cloud_timed):
        # initialzation
        # y is the exist latency of one of the cloud
        U_min = U.min(axis=0)
        L_max = L.max(axis=0)
        # U[0]={0}, L[0]={0}, the tick start from 1
        tick += 1
        for cloud_id in range(self.N):
            latency_cloud = latency_cloud_timed[:,cloud_id]
            latency_cloud_exist = np.delete(latency_cloud, np.where(latency_cloud == 0))
            U[tick][cloud_id] = (tick - 1) / tick * U[tick - 1][cloud_id] + (latency_cloud_exist[-1] - np.average(latency_cloud_exist) - self.δ)
            # U_changed is array of bools, like [False, False, True, True, False, False]
            U_changed = U[tick, :] - U_min >= self.b_increase
            # U_changed_ticks is array of ticks, like [0, 0, 5, 6, 0, 0]
            U_changed_ticks = np.array([index if changed else 0 for index, changed in enumerate(U_changed)])
            logger.info(f'tick: {tick}, U_changed: {U_changed}, U_changed_ticks: {U_changed_ticks}')
            L[tick][cloud_id] = (tick - 1) / tick * L[tick - 1][cloud_id] + (latency_cloud_exist[-1] - np.average(latency_cloud_exist) + self.δ)
            L_changed = L_max -L[tick, :] >= self.b_decrease
            L_changed_ticks = np.array([index if changed else 0 for index, changed in enumerate(L_changed)])
            logger.info(f'tick: {tick}, L_changed: {L_changed}, L_changed_ticks: {L_changed_ticks}')
            changed = U_changed + L_changed
            changed_ticks = U_changed_ticks + L_changed_ticks
            return changed, changed_ticks
        
        
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
            header = ['timestamp', 'file_id', 'file_size', 'file_read', 'latency', 'placement_policy']
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
    algorithm = AW_CUCB(data, file_metadata)
    algorithm.processing()
    logger.info(f'processing finished')
    algorithm.save_result()
    logger.info(f'save_result finished')
    
if __name__ == "__main__":
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
