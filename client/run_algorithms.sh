#!/usr/bin/env bash

# check if the user has provided the correct number of arguments
# one argument file_path should be provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 file_path"
    exit 1
fi

# check if the file exists
if [ ! -f $1 ]; then
    echo "File $1 does not exist"
    exit 1
fi

# try to build metadata for the input file in blocking mode
echo "Building metadata for $1"
python $(dirname "$0")/algorithm.py --only_preprocess True $1
echo "Metadata built"

# run the four algorithms in parallel in background
echo "Running algorithms"
python $(dirname "$0")/algorithm.py $1 &
python $(dirname "$0")/algorithm_simple.py $1 &
python $(dirname "$0")/algorithm_ewma.py $1 &
python $(dirname "$0")/algorithm_random.py $1 &
echo "Algorithms running in background, use ps -ef | grep python to check their status"
