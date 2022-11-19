from datetime import datetime
import time
from halo import Halo
import fire
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy


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

def process(file_input: str = 'client/test.txt', file_output: str = 'processed.txt'):
    # calculate the file lines of the file_input
    with spinner_context('Calculate file line count ...') as spinner:
        file_input_lines = get_file_line_count(file_input)
    print(f'file_input: {file_input} count: {file_input_lines}')
    
    lines = []
    # process the file_input
    with spinner_context(f'Processing {file_input} ...') as spinner, open(file_input) as fin:
        # update the spinner text to show the progress in 00.01% minimum
        update_tick = int(file_input_lines / 10000 or 100)
        for index, line in enumerate(fin):
            # insert the parsed human readable datetime string of the first column at index 1
            # the last column contains the newline character, keep it
            line_segements = line.split(',')
            try:
                datetime_str = datetime.fromtimestamp(
                    int(line_segements[0]) / 10 ** 8).strftime('%Y-%m-%d %H:%M:%S.%f')
                line_segements.insert(1, datetime_str)
                lines.append(line_segements)
            except Exception as e:
                print(
                    f'Error parsing line {index}: {line}, line_segements: {line_segements}, skip')
                continue
            if index % update_tick == 0:
                spinner.text = f'Processing {index / file_input_lines * 100:.2f}%'
    print(f"processed file_input: {file_input}, {len(lines)} lines")
    
    # Sort the lines by the timestamp
    with spinner_context('Sort the lines ...'):
        # sort the lines by the first column (timestamp, index 0), then the fifth column (Read/Write index 4) in descending order
        # https://iditect.com/faq/python/how-to-sort-objects-by-multiple-keys.html#How%20to%20sort%20a%20list%20with%20two%20keys%20but%20one%20in%20reverse%20order?
        class reversor:
            def __init__(self, obj):
                self.obj = obj
            def __eq__(self, other):
                return other.obj == self.obj
            def __lt__(self, other):
                return other.obj < self.obj
        lines.sort(key=lambda line: (line[0], reversor(line[4])))
    print(f"sorted {len(lines)} lines")
    
    # cut the lines to 10 in each second, after sorting
    limits_per_second = 10
    with spinner_context(f'Limiting lines to {limits_per_second} in each second'):
        lines = limit_lines_by_timestamp(lines, limits_per_second)
    print(f'After limiting, {len(lines)} lines left')
    
    # Save the processed lines to the file_output
    with spinner_context('Saving the results ...'), open(file_output, 'w') as fout:
        for line in lines:
            fout.write(','.join(line))
    print(f'Saved to: {file_output}')

def limit_lines_by_timestamp(lines, limit=10):
    # use pandas groupby to group the lines by the timestamp in seconds
    df = pd.DataFrame(lines)
    df['timestamp_in_seconds'] = df.apply(lambda x: x[0][:10], axis=1)
    df_timestamp_group: DataFrameGroupBy = df.groupby('timestamp_in_seconds')
    # TODO: this method no response, need to fix
    # for _, group in df_timestamp_group:
    #     df.drop(group[limit:].index, inplace=True)
    # return df.values.tolist()
    df_result = DataFrame()
    for _, group in df_timestamp_group:
        df_result = pd.concat([df_result, group.head(limit)])
    df_result.drop(columns=['timestamp_in_seconds'], inplace=True)
    return df_result.values.tolist()
    

if __name__ == "__main__":
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(process)
