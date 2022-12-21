#!/usr/bin/env python3
# coding=utf-8

import fire
import tempfile
from os import urandom
from os.path import exists, join


def main(file_path: str = None, size: int = 1024 * 1024 * 10):
  # If the file path is not specified or exists, use /tmp/<size>.tmp as the default file path
  if not file_path or not exists(file_path):
    file_path = join(tempfile.gettempdir(), f'{size}.tmp')
  with open(file_path, 'wb') as fout:
    fout.write(urandom(size))
  print(f'File {file_path} generated with size {size} bytes.')


if __name__ == "__main__":
  fire.core.Display = lambda lines, out: print(*lines, file=out)
  fire.Fire(main)
