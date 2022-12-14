#!/usr/bin/env python

import fire
import random
from os.path import splitext, exists
from textwrap import dedent


def main(input_file: str = 'web_2_sized_50000_55000_wp_ration_9.txt', output_file: str = None, line_num: str = '', copies: int = 10, copies_minmum: int = 1, random_copy: bool = True, shuffle: bool = True, verbose: bool = True):
  """Duplicate lines from the input file, and write to the output file.

  Args:
      input_file (str): The path to the file to be duplicated.
      output_file (str, optional): The path to the output file. Defaults to input_file_processed.
      line_num (str): The line number to be duplicated. Defaults to ''. Support multiple lines, separated by comma or line number range, separated by dash.
      copies (int, optional): The number of copies. Defaults to 10.
      copies_minmum (int, optional): The minimum number of copies. Defaults to 1, only effective when random_copy is True.
      random_copy (bool): Wether to dumplicate the lines random_copy copies from 0 to copies. Defaults to True.
      shuffle (bool): Wether to shuffle the lines of file. Defaults to True.
      verbose (bool, optional): Whether to print verbose progress to the console. Defaults to True.
  """

  if not exists(input_file):
    raise FileNotFoundError(f"File {input_file} not found.")

  if not output_file:
    file_name, file_extension = splitext(input_file)
    output_file = f'{file_name}_processed{file_extension}'
  # parse the line_num
  parsed_line_num = []
  if line_num:
    # line_num maybe a string or a tuple like (1, 2, 3)
    line_num_segements = line_num.split(',') if type(line_num) == str else map(str, line_num) if type(line_num) == tuple else line_num
    for line_num_segement in line_num_segements:
      if '-' in line_num_segement:
        start, end = line_num_segement.split('-')
        parsed_line_num.extend(range(int(start), int(end) + 1))
      else:
        parsed_line_num.append(int(line_num_segement))
    # remove duplicates of parsed_line_num
    parsed_line_num = sorted(list(set(parsed_line_num)))    
  if verbose:
    info = dedent(f'''
      input_file: {input_file}
      output_file: {output_file}
      line_num: {line_num}, parsed: {parsed_line_num}
      copies: {copies}
      random_copy: {random_copy}
      shuffle: {shuffle}
    ''')
    print(info)
  with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    lines = fin.readlines()
    file_lines = len(lines)
    print(f'input file {input_file} has {file_lines} lines')
    lines_to_copy = []
    for line_num in parsed_line_num:
      line: str = lines[line_num]
      if random_copy:
        copies = random.randint(copies_minmum, copies + 1)
      lines_to_copy.append((line, copies, line_num))
    # copy line, insert after the index of the line
    # the location of the insert position may be changed after the insert
    for line, copies, line_num in lines_to_copy:
      print(f'insert {copies} copies of {line.strip()} after index {line_num}')
      insert_lines(lines, line, line_num, copies)
    # shuffle the lines
    if shuffle:
      print(f'shuffling {len(lines)} lines...')
      random.shuffle(lines)
    print(f'saving to {output_file}, {len(lines)} lines...')
    fout.writelines(lines)


def insert_lines(lines: list[str], line: str, start: int, copies: int):
  # get random insert locations, sort it
  locations = sorted(random.choices(range(start, len(lines)), k=copies))
  print(f'---> copy to {locations}, selected from {start} to {len(lines)}')
  for location in locations:
    lines.insert(location, line)


if __name__ == "__main__":
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
