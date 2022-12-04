#!/usr/bin/env python3

from dataclasses import dataclass
import sys
import os
from os.path import dirname, join, realpath
import itertools
import random

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
    # timestamp in seconds
    timestamp: int
    file_id: int
    offset: int
    file_size: int
    file_read: bool = True
    datetime_offset: int = 0
    latency: int = -1
    latency_full: int = -1
    placement_policy: list[int] = None
    migration_targets: list[int] = None
    LB: list[int] = None
    eit: list[int] = None
    u_hat_it: list[int] = None
    post_reward: float = 0
    post_cost: float = 0
    migration_gains: float = 0
    migration_cost: float = 0
    
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
                # size = 1024 * 1024 * 10
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

def float_to_string(x):
    return f'{x:.8f}' if x != 0 else '0'