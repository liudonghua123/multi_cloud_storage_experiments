#!/usr/bin/env python3
# coding=utf-8

import fire
import csv
from datetime import datetime

def main(input_file: str = 'input.csv', output_file: str = 'output.csv', timestamp_column_name: str = "timestamp", to_readable_timestamp: bool = True, datetime_format: str = '%Y-%m-%d %H:%M:%S.%f'):
  # get the first line of the input file, which is the header
  with open(input_file, 'r') as fin:
    reader = csv.reader(fin)
    headers = next(reader)
  # newline='' for open output file is for avoiding blank lines between rows
  # more details: https://docs.python.org/3/library/csv.html#id3
  # https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
  with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=headers)
    writer.writeheader()
    print(f'headers having {len(headers)} columns: {headers}')
    rows = []
    for row in reader:
      if to_readable_timestamp:
        # convert timestamp to readable timestamp
        row[timestamp_column_name] = datetime.fromtimestamp(
          int(row[timestamp_column_name])).strftime(datetime_format)
      else:
        # convert readable timestamp to timestamp
        row[timestamp_column_name] = datetime.strptime(
          row[timestamp_column_name], datetime_format).timestamp()
      rows.append(row)
    print(f'rows having {len(rows)} rows, fist 5 rows: {rows[:5]}')
    writer.writerows(rows)

if __name__ == "__main__":
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
