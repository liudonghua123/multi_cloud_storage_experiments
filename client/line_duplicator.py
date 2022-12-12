#!/usr/bin/env python

import fire
import random
from os.path import splitext, exists
from textwrap import dedent


def main(file_input: str = 'web_2_sized_50000_55000_wp_ration_9.txt', file_output: str = None, line_num: str = '', copies: int = 10, copies_minmum: int = 1, random: bool = True, verbose: bool = True):
  """Duplicate lines from the input file, and write to the output file.

  Args:
      file_input (str): The path to the file to be duplicated.
      file_output (str, optional): The path to the output file. Defaults to file_input_processed.
      line_num (str): The line number to be duplicated. Defaults to ''. Support multiple lines, separated by comma or line number range, separated by dash.
      copies (int, optional): The number of copies. Defaults to 10.
      copies_minmum (int, optional): The minimum number of copies. Defaults to 1, only effective when random is True.
      random (bool): Wether to dumplicate the lines random copies from 0 to copies. Defaults to True.
      verbose (bool, optional): Whether to print verbose progress to the console. Defaults to True.
  """

  if not exists(file_input):
    raise FileNotFoundError(f"File {file_input} not found.")

  if not file_output:
    file_name, file_extension = splitext(file_input)
    file_output = f'{file_name}_processed{file_extension}'
  # parse the line_num
  parsed_line_num = []
  if line_num:
    line_num_segements = line_num.split(',')
    for line_num_segement in line_num_segements:
      if '-' in line_num_segement:
        start, end = line_num_segement.split('-')
        parsed_line_num.extend(range(int(start), int(end) + 1))
      else:
        parsed_line_num.append(int(line_num_segement))
  if verbose:
    info = dedent(f'''
      file_input: {file_input}
      file_output: {file_output}
      line_num: {line_num}, parsed: {parsed_line_num}
      copies: {copies}
      random: {random}
    ''')
    print(info)
  with open(file_input, 'r') as fin, open(file_output, 'w') as fout:
    lines = fin.readlines()
    file_lines = len(lines)
    print(f'input file {file_input} has {file_lines} lines')
    for line_num in parsed_line_num:
      line: str = lines[line_num]
      if random:
        copies = random.randint(copies_minmum, copies + 1)
      print(f'insert {copies} copies of {line.strip()} after index {line_num}')
      insert_lines(lines, line, line_num, copies, file_lines)
    print(f'saving to {file_output}, {len(lines)} lines...')
    fout.writelines(lines)


def insert_lines(lines: list[str], line: str, start: int, copies: int, file_lines: int):
  # get random insert locations
  locations = random.choices(range(start, file_lines), k=copies)
  print(f'---> copy to {locations}')
  for location in locations:
    lines.insert(location, line)


if __name__ == "__main__":
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
