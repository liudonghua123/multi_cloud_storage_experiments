#!/usr/bin/env python

from datetime import datetime
import time
from halo import Halo
import fire
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
from os.path import splitext


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

def get_read_write_count(lines: list[list[str]], operation_field_index: int = 4):
    read_count = 0
    write_count = 0
    for line in lines:
        if line[operation_field_index] == 'Read':
            read_count += 1
        else:
            write_count += 1
    return read_count, write_count

class reversor:
    def __init__(self, obj):
        self.obj = obj
    def __eq__(self, other):
        return other.obj == self.obj
    def __lt__(self, other):
        return other.obj < self.obj
class spinner_context:
    def __init__(self, start_text: str, end_text: str = None, spinner_indicator: str = 'dots'):
        self.start_text = start_text
        self.end_text = end_text or start_text
        self.spinner_indicator = spinner_indicator
        self.spinner = Halo(text=self.start_text, spinner=self.spinner_indicator)
        self.start_time = None
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.spinner.start()
        return self.spinner
    def __exit__(self, exc_type, exc_value, traceback):
        self.spinner.succeed(f'{self.end_text}, took {time.perf_counter() - self.start_time:.2f}s')


def process(file_input: str = 'test.txt', file_output: str = None, limit: bool = False, limit_lower: int = 10, limit_upper: int = 100, limit_percent: float = 0.1, size_control: bool = True, size_lower: int = 50 * 1024, size_upper: int = 100 * 1024, rw_ration: float = 0, add_timestamp: bool = True, sort_by_timestamp_and_write: bool = False, sort_by_write_and_timestamp: bool = True):
    
    """
    Process the input file and output the result to the output file. 
    READ->PROCESS[->LIMIT][->SIZE_CONTROL][->RW_RATION][->SORT][->ADD_HUMNAN_READABLE_TIMESTAMP]->WRITE

    This function will read the input file line by line, and process each line, insert a hunman readable timestamp.
    Then sort the lines by the timestamp, and limit the lines by the timestamp if the limit > 0 
    (the default limit 0 means do not apply limit operation) is specified.
    And Finally, write the result to the output file.

    Parameters:
    file_input (str): the input file path
    file_output (str): the output file path
    limit (bool): whether to apply limit operation
    limit_lower (int): the limit lower bound for the amount of lines in each second
    limit_upper (int): the limit upper bound for the amount of lines in each second
    limit_percent (float): the limit percent for the amount of lines in each second
    size_control (bool): whether to apply size control operation
    size_lower (int): the size lower bound for data to filter 
    size_upper (int): the size upper bound for data to filter
    rw_ration (float): the ratio of read/write, 0 means no filter
    add_timestamp (bool): whether to add timestamp to the output file
    sort_by_timestamp_and_write (bool): whether to sort the output file by the timestamp then the write/read
    sort_by_write_and_timestamp (bool): whether to sort the output file by the write/read then the timestamp
    
    Returns:
    None

    """
    
    # normalize the arguments, if some number arguments are passed as expression, evaluate them
    def normalize_value(arg):
        result = arg
        if isinstance(arg, str):
            result = eval(arg)
            if isinstance(result, float): 
                result = float(f'{result:.2f}')
        elif isinstance(arg, float):
            result = float(f'{arg:.2f}')
        return result
    
    limit_lower, limit_upper, limit_percent, size_lower, size_upper, rw_ration = map(normalize_value, [limit_lower, limit_upper, limit_percent, size_lower, size_upper, rw_ration])        
    
    print(f"Input file: {file_input}, output file: {file_output}, limit: {limit}, limit_lower: {limit_lower}, limit_upper: {limit_upper}, limit_percent: {limit_percent}, size_control: {size_control}, size_lower: {size_lower}, size_upper: {size_upper}, add_timestamp: {add_timestamp}, sort_by_timestamp_and_write: {sort_by_timestamp_and_write}, sort_by_write_and_timestamp: {sort_by_write_and_timestamp}")
    
    # calculate the file lines of the file_input
    with spinner_context('Calculate file line count ...') as spinner:
        file_input_lines = get_file_line_count(file_input)
    print(f'file_input: {file_input} count: {file_input_lines}')
    
    lines = []
    
    has_human_readable_timestamp = False
    
    # process the file_input, save the splited lines to the lines list
    with spinner_context(f'Processing {file_input}, saveing into lines list ...') as spinner, open(file_input) as fin:
        # update the spinner text to show the progress in 00.01% minimum
        update_tick = int(file_input_lines / 1000 if file_input_lines > 1000 else 100)
        for index, line in enumerate(fin):
            # insert the parsed human readable datetime string of the first column at index 1
            # the last column contains the newline character, keep it
            lines.append(line.split(','))
            if index % update_tick == 0:
                spinner.text = f'Processing {index / file_input_lines * 100:.2f}%'
        # use the last line to check if the file_input has human readable timestamp
        has_human_readable_timestamp = len(line.split(',')) > 7
    print(f'detected has_human_readable_timestamp: {has_human_readable_timestamp}')
    print(f"processed file_input: {file_input}, {len(lines)} lines")
    
    # calculate the operation_field_index according to the has_human_readable_timestamp
    operation_field_index = 4 if has_human_readable_timestamp else 3
    
    # Cut the lines to limit in each second, after sorting
    if limit:
        with spinner_context(f'Limiting lines to limit_lower: {limit_lower}, limit_upper: {limit_upper}, limit_percent: {limit_percent} in each second'):
            lines = limit_lines_by_timestamp(lines, limit_lower, limit_upper, limit_percent)
        print(f'After limiting, {len(lines)} lines left')
        
    # Filter by size
    if size_control:
        with spinner_context(f'Filtering lines by size size_lower: {size_lower}, size_upper: {size_upper}'):
            lines = filter_lines_by_size(lines, size_lower, size_upper)
        print(f'After filtering size, {len(lines)} lines left')
    
    # filter by rw_ratio
    if rw_ration:
        with spinner_context(f'Filtering lines by rw_ration: {rw_ration}'):
            lines = filter_lines_by_rw_ratio(lines, rw_ration, operation_field_index)
        print(f'After filtering rw_ratio, {len(lines)} lines left')
            
    # Sort the lines by the timestamp
    if sort_by_timestamp_and_write:
        with spinner_context('Sort the lines ...'):
            # sort the lines by the first column (timestamp, index 0), then the fifth column (Read/Write index 4) in descending order
            # https://iditect.com/faq/python/how-to-sort-objects-by-multiple-keys.html#How%20to%20sort%20a%20list%20with%20two%20keys%20but%20one%20in%20reverse%20order?
            lines.sort(key=lambda line: (line[0], reversor(line[operation_field_index])))
        print(f"sorted {len(lines)} lines")
    
    # Sort the lines by the write
    if sort_by_write_and_timestamp:
        with spinner_context('Sort the lines ...'):
            lines.sort(key=lambda line: (reversor(line[operation_field_index]), line[0]))
        print(f"sorted {len(lines)} lines")
    
    # Add human readable timestamp
    if add_timestamp:
        with spinner_context('Adding hunman readable timestamp ...'):
            for line in lines:
                # check whether the line has the hunman readable timestamp already
                if has_human_readable_timestamp:
                    print(f'file {file_input} first line {line} already has timestamp, skip this process')
                    break
                try:
                    datetime_str = datetime.fromtimestamp(
                        int(line[0]) / 10 ** 8).strftime('%Y-%m-%d %H:%M:%S.%f')
                    line.insert(1, datetime_str)
                except Exception as e:
                    print(
                        f'Error parsing line {index}: {line}, skip')
                    continue
            has_human_readable_timestamp = True
            # update operation_field_index
            operation_field_index = 4 if has_human_readable_timestamp else 3
        
    # Report the read and write count
    with spinner_context('Counting the read/write ...'):
        read_count , write_count = get_read_write_count(lines, operation_field_index)
    print(f"read_count: {read_count}, write_count: {write_count}")
    
    # WRITING...
    # If the output file is not specified, then use the input file name with the some operation suffix
    if not file_output:
        file_input_name, file_input_extension = splitext(file_input)
        file_output = f'{file_input_name}'
        if limit:
            file_output += f'_limit_{limit_lower}_{limit_upper}_{limit_percent}'
        if size_control:
            file_output += f'_size_{size_lower}_{size_upper}'
        if rw_ration:
            file_output += f'_rw_{rw_ration}'
        if sort_by_timestamp_and_write:
            file_output += f'_sort_t_w'
        if sort_by_write_and_timestamp:
            file_output += f'_sort_w_t'
        if add_timestamp:
            file_output += f'_timestamped'
        file_output += file_input_extension
    # Save the processed lines to the file_output
    with spinner_context('Saving the results ...'), open(file_output, 'w') as fout:
        for line in lines:
            fout.write(','.join(line))
    print(f'Saved to: {file_output}') 

def filter_lines_by_rw_ratio(lines, rw_ration, operation_field_index=4):
    read_count , write_count = get_read_write_count(lines, operation_field_index)
    # For example, if read_count is 100, write_count is 100, and rw_ration is 2, then we need to use 100 read and 50 write.
    # For the same example rw_ration is 0.5, then we need to use 50 read and 100 write.
    # 100(read)/2(rw_ration) = 50(write) < 100(write).
    if read_count / rw_ration < write_count:
        print(f'rw_ratio: {rw_ration}, read_count: {read_count}, write_count: {write_count}, write is more then read according to the ratio, filter out some write')
        actural_read = read_count
        actural_write = int(actural_read / rw_ration)
    else:
        print(f'rw_ratio: {rw_ration}, read_count: {read_count}, write_count: {write_count}, read is more then write according to the ratio, filter out the read')
        actural_write = write_count
        actural_read = int(actural_write * rw_ration)
    # Filter the lines by the read/write ratio
    results = []
    for line in lines:
        if line[operation_field_index] == 'Read':
            if actural_read > 0:
                results.append(line)
                actural_read -= 1
        else:
            if actural_write > 0:
                results.append(line)
                actural_write -= 1
    return results

def filter_lines_by_size(lines, size_lower, size_upper):
    # use pandas to filter the lines by size
    # https://www.geeksforgeeks.org/ways-to-filter-pandas-dataframe-by-column-values/
    # https://www.geeksforgeeks.org/filter-pandas-dataframe-with-multiple-conditions/
    # dataFrame[(dataFrame['Salary']>=100000) & (dataFrame['Age']<40) & dataFrame['JOB'].str.startswith('P')][['Name','Age','Salary']]
    df = pd.DataFrame(lines)
    print(f'\ndf.head(): {df.head()}, df.shape: {df.shape}, len(lines): {len(lines)}\n')
    # the size is in the 7th column, index 6
    df = df[(df[6].astype(int) >= size_lower) & (df[6].astype(int) <= size_upper)]
    return df.values.tolist()

def limit_lines_by_timestamp(lines, limit_lower, limit_upper, limit_percent):
    # use pandas groupby to group the lines by the timestamp in seconds
    df = pd.DataFrame(lines)
    df['timestamp_in_seconds'] = df.apply(lambda x: x[0][:10], axis=1)
    df_timestamp_group: DataFrameGroupBy = df.groupby('timestamp_in_seconds')
    # use this method to speed up the drop operation
    drop_index = []
    for _, group in df_timestamp_group:
        limit = min(max(int(len(group) * limit_percent), limit_lower), limit_upper)
        # group.index[limit:].tolist() is the same as group[limit:].index.tolist()
        drop_index += group.index[limit:].tolist()
    df.drop(drop_index, inplace=True)
    df.drop(columns=['timestamp_in_seconds'], inplace=True)
    return df.values.tolist()
    # df_result = DataFrame()
    # for _, group in df_timestamp_group:
    #     df_result = pd.concat([df_result, group.head(limit)])
    # df_result.drop(columns=['timestamp_in_seconds'], inplace=True)
    # return df_result.values.tolist()
    

if __name__ == "__main__":
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(process)
