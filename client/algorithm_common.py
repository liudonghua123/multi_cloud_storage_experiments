#!/usr/bin/env python3

from dataclasses import dataclass
import sys
import os
from os.path import dirname, join, realpath, basename, exists
import itertools
import random
import numpy as np

# serialize and deserialize using jsonpickle or pickle
# jsonpickle is better for human readable
# pickle is better for performance and smaller size
import pickle
from typing import Literal # pickle.dumps / pickle.loads

# In order to run this script directly, you need to add the parent directory to the sys.path
# Or you need to run this script in the parent directory using the command: python -m client.algorithm
sys.path.append(dirname(dirname(realpath(__file__))))

from common.utility import spinner_context, get_config, init_request_retries_session
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
max_retries: int = config['max_retries']

init_request_retries_session(cloud_providers, max_retries)

@dataclass
class FileMetadata:
    offset: int
    size: int
    placement: list[int] = None
    
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
    

class ChangePoint:
    INCREASE = 'increase'
    DECREASE = 'decrease'
    
    def __init__(self, tick: int, type: Literal["increase", "decrease"]) -> None:
        self.tick = tick
        self.type = type
    
    def __add__(self, other):
        if other is None:
            return self
        raise RuntimeError("ChangePoint can only be added to None")
    
    def __str__(self) -> str:
        return f"ChangePoint(tick={self.tick}, type={self.type})"
    
    def __repr__(self) -> str:
        return f"ChangePoint(tick={self.tick}, type={self.type})"
    
@dataclass
class TraceData:
    # if the tick is -1, it means the data is not initialized
    tick: int = -1
    # timestamp in seconds
    timestamp: int = 0
    file_id: int = 0
    offset: int = 0
    file_size: int = 0
    file_read: bool = True
    datetime_offset: int = 0
    latency: int = -1
    latency_full: int = -1
    placement: list[int] = None
    placement_policy: list[int] = None
    migration_targets: list[int] = None
    LB: list[int] = None
    eit: list[int] = None
    u_hat_it: list[int] = None
    post_reward: float = 0
    post_cost: float = 0
    migration_path: str = ''
    migration_gains: float = 0
    migration_cost: float = 0
    U: str = ''
    L: str = ''
    U_min: str = ''
    L_max: str = ''
    changed_ticks_trace: str = ''
    request_datetime: str = ''
    latency_accumulated_average: int = -1
    post_reward_accumulated_average: float = 0
    post_cost_accumulated_average: float = 0
    u_hat_it_accumulated_average : list[float] = None
    post_cost_accumulation: float = 0
    window_sizes : list[int] = None
    last_change_tick : list[int] = None
    eit_trace: str = ''
    
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
        # update default_data_file and default_metadata_file, use basename of file_path as prefix
        prefix = basename(file_path).split('.')[0]
        self.default_data_file = f'{prefix}_trace_data.bin'
        self.default_metadata_file = f'{prefix}_metadata.bin'
    
    def load_data(self):
        if exists(self.default_data_file) and exists(self.default_metadata_file):
            logger.info(f'using the serilized data')
            return self._load_data_via_deserialize()
        logger.info(f'parse {self.file_path} data')
        data, file_metadata = self._parse_data()
        logger.info(f'serialize the parsed to {self.default_data_file} and {self.default_metadata_file}')
        self._save_data_via_serialize(data, file_metadata)
        return data, file_metadata
    
    def _load_data_via_deserialize(self, data_file: str = None, file_metadata_file: str = None):
        with open(data_file or self.default_data_file, 'rb') as data_file_fd, open(file_metadata_file or self.default_metadata_file, 'rb') as file_metadata_file_fd:
            data = pickle.load(data_file_fd)
            file_metadata = pickle.load(file_metadata_file_fd)
        return data, file_metadata
    
    def _save_data_via_serialize(self, data, file_metadata, data_file: str = None, file_metadata_file: str = None):
        with open(data_file or self.default_data_file, 'wb') as data_file_fd, open(file_metadata_file or self.default_metadata_file, 'wb') as file_metadata_file_fd:
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
                # size = 1024 * 1024 * 10
                read = operation_type == 'Read'
                initial_optimized_placement = list(itertools.combinations(range(self.N), self.n))
                # For the first len(initial_optimized_placement) trace data, overwrite write operation ignore the same file_id
                # Not need metadata info
                if index < len(initial_optimized_placement):
                    read = False
                # For the rest trace data, only set metadata for read operation.
                elif read:
                    # create file_metadata if does not exists
                    if self.file_metadata.get(file_id) is None:
                        # 1. The placement for read will be randomly selected
                        placement = random.choice(initial_optimized_placement)
                        placement = [1 if i in placement else 0 for i in range(self.N)]
                        logger.info(f'set file_metadata.placement: ${placement} for file_id: {file_id}, read: {read}')
                        self.file_metadata[file_id] = FileMetadata(offset, size, placement)
                        
                self.data.append(TraceData(timestamp=timestamp, file_id=file_id, offset=offset, file_size=size, file_read=read, datetime_offset=timestamp - self.initial_timestamp))
                if index % update_tick == 0:
                    spinner.text = f'Processing {index / self.file_input_lines * 100:.2f}%'
        return self.data, self.file_metadata

def float_to_string(x):
    return f'{x:.8f}' if x != 0 else '0'

def min_except_zero(x):
    '''
    x: list, tuple, or numpy array
    Find the minimum value in x, except 0
    '''
    return min([i for i in x if i != 0])

def max_except_zero(x):
    '''
    x: list, tuple, or numpy array
    Find the maximum value in x, except 0
    '''
    return max([i for i in x if i != 0])

def argmin_except_zero(x):
    '''
    x: list, tuple, or numpy array
    Find the original index of the minimum value in x from the end to the start position, except 0
    '''
    return np.where(x == min_except_zero(x))[0][-1]

def argmax_except_zero(x):
    '''
    x: list, tuple, or numpy array
    Find the original index of the maximum value in x from the end to the start position, except 0
    '''
    return np.where(x == max_except_zero(x))[0][-1]

def find_window_sized_index(windows_size, x, changed_tick = 0):
    '''
    windows_size: int, minimum value is 1
    x: list, tuple, or numpy array
    Find the first index of the in x execlude zeros from the end to the windows_size position
    eg: 
    find_window_sized_index(5,[1,0,0,0,0,0,1,1]) -> 0
    find_window_sized_index(5,[0,0,0,0,0,1,1]) -> 5
    find_window_sized_index(5,[1,0,0,0,0,0,1,1,0,1,0,1,1,1]) -> 7
    find_window_sized_index(5,[1,0,0,0,0,0,1,1,0,1,0,1,1,1],7) -> 7
    find_window_sized_index(5,[1,0,0,0,0,0,1,1,0,1,0,1,1,1],9) -> 8
    '''
    length = len(x)
    result_index = length - 1
    processed_count = 0
    for i, v in enumerate(reversed(x)):
        reversed_index = length - i - 1
        if x[reversed_index] != 0 and reversed_index >= changed_tick:
            result_index = reversed_index
            processed_count += 1
            if processed_count >= windows_size:
                break
    return result_index

def calculate_accumulated_average(x):
    '''
    x: list, tuple, or numpy array
    Calculate the accumulated average of x
    '''
    return (np.cumsum(x) / np.arange(1, len(x) + 1)).tolist()

def calculate_accumulated_average_matrix(x):
    '''
    x: two demensional list, tuple, or numpy array
    Calculate the accumulation of x
    '''
    return np.cumsum(x, axis=0).tolist()

def calculate_accumulation(x):
    '''
    x: list, tuple, or numpy array
    Calculate the accumulation of x
    '''
    return np.cumsum(x).tolist()