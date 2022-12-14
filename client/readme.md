# client

The client of adaptive data placement in multi-cloud storage.

### network_test

Make http request to the clould providers, the request is configured via the product of  `network_test_visualization`, `data_sizes` and `read` in `network_test.matrix` section of `config.yml`. Some other configurations is under the rest `network_test` section.

`network_test_thread.py` is a rewrite work of `network_test.py`, use *`multi-thread`* instead of *`coroutines`* for performance and control more precisely.

#### How to run

1. configure the `config.yml`
2. cd client
3. pip install -r requirements.txt
4. python network_test_thread.py

### algorithm

There are 4 algorithms in this directory, `algorithm.py` (aw_cucb, also means main), `algorithm_simple.py`, `algorithm_ewma.py` and `algorithm_random.py`. All of them accept a `input_file` argument, you can view the detailed usage via `python algorithm[_simple][_ewma][_simple].py --help`.

Also notice, there are two special arguments for the main `algorithm.py`.

- only_preprocess, default False, if set True, only metadata construction is processed.
- migration, default False, if set True, the migration step will skip for aw_cucb algorithm.

You can also run all the 4 algorithms parallel via `run_algorithms.sh` or compare aw_cucb and aw_cucb_no_migration via `run_algorithms_migration.sh`. The process time may vary depend on the size trace_data. Use `screen`(recommend) or `nohup` to run these scripts.

All the results are saved in `result_<trace_file_name>`. And you can also use `./algorithm_visualization.py` to visualize the results.

#### How to run

1. run individual algorithm via `python algorithm[_simple][_ewma][_simple].py /path/to/trace_data`
2. run all the algorithms via `./run_algorithms.sh`.
3. run migration comparation via `./run_algorithms_migration.sh`.

### network_test_visualization

Simple visualize the data generated by the previous `network_test`.

If plain network_test results csv file is provided, the aggregated data will be calculated, save as csv file in `network_test_results` directory of cwd, then visualize it via matplotlib. The latter visualization can use the aggregated csv file directly without repeated calculation.

#### how to run

1. run the previous `network_test` to generate the csv result file.
2. run `python network_test_visualization.py`, you can add `--cloud_ids` and(or) `--aggregations` arguments for custom visualization, for example, `--cloud_ids cloud_id_0,cloud_id_1 --aggregations min,mean,max,p50`.
3. run `python network_test_visualization.py --help` for detailed usage.

The cli help usage.

```shell
D:\code\python\multi_cloud_storage_experiments\client>python network_test_visualization.py  --help
INFO: Showing help with the command 'network_test_visualization.py -- --help'.

NAME
    network_test_visualization.py - Visualize the network test aggregated results.

SYNOPSIS
    network_test_visualization.py <flags>

DESCRIPTION
    If network_test_aggregated_results_csv_file_path provided, then use the date of csv file to visualize.
    Else, use the network_test_results_csv_file_path to calculate the aggregated results and visualize.

FLAGS
    --network_test_results_csv_file_path=NETWORK_TEST_RESULTS_CSV_FILE_PATH
        Type: str
        Default: 'network_test_results/network_test_with_placements_1_1_1_1_d...
        the csv file path of network test results.
    --network_test_aggregated_results_csv_file_path=NETWORK_TEST_AGGREGATED_RESULTS_CSV_FILE_PATH
        Type: str
        Default: 'network_test_results/network_test_aggregated__2022_11_02_17...
        the csv file path of network test aggregated results.
    --cloud_ids=CLOUD_IDS
        Type: str
        Defa...
        the cloud ids to visualize, separated by comma.
    --aggregations=AGGREGATIONS
        Type: str
        Default: ['p50', 'p99']
        the aggregations to visualize, separated by comma.
    --legend_loc=LEGEND_LOC
        Type: str
        Default: 'upper right'
        the location of legend, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html for more details.        

D:\code\python\multi_cloud_storage_experiments\client
```

### Some helpful links

- https://requests.readthedocs.io/en/latest/
- https://realpython.com/python-requests/
- https://realpython.com/async-io-python/
- https://skipperkongen.dk/2016/09/09/easy-parallel-http-requests-with-python-and-asyncio/
- https://github.com/encode/httpx
- https://www.python-httpx.org/async/
- https://github.com/yaml/pyyaml/
- https://python.land/data-processing/python-yaml
- https://www.geeksforgeeks.org/python-schedule-library/
- https://fedingo.com/how-to-schedule-task-in-python/
- https://www.cuemath.com/numbers/math-symbols/
- https://byjus.com/maths/math-symbols/
- https://realpython.com/python-enumerate/
- https://pythonguides.com/python-list-comprehension-using-if-else/
- https://linuxhint.com/python-randomly-select-from-list/
- https://www.geeksforgeeks.org/numpy-minimum-in-python/
- https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
- https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
- https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html
- https://datascienceparichay.com/article/sort-numpy-array-in-descending-order/
- https://www.nickmccullum.com/advanced-python/numpy-indexing-assignment/
- https://pyneng.readthedocs.io/en/latest/book/19_concurrent_connections/concurrent_futures_submit.html
- https://realpython.com/intro-to-python-threading/#join-a-thread
- https://coderslegacy.com/python/get-return-value-from-thread/
- https://www.delftstack.com/howto/python/python-pool-map-multiple-arguments/
- https://python-tutorials.in/python-asyncio-gather/
- https://hackernoon.com/how-to-run-asynchronous-web-requests-in-parallel-with-python-3-5-without-aiohttp-264dc0f8546
- https://google.github.io/python-fire/guide/
- https://github.com/google/python-fire
- https://github.com/google/python-fire/issues/188#issuecomment-1165098568
- https://www.geeksforgeeks.org/python-docstrings/
- https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
- https://pythonguides.com/matplotlib-set_xticklabels/
- https://www.delftstack.com/zh/howto/matplotlib/how-to-rotate-x-axis-tick-label-text-in-matplotlib/
- https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
- https://www.askpython.com/python/examples/error-bars-in-python
- https://iditect.com/faq/python/how-to-sort-objects-by-multiple-keys.html
- https://www.autoscripts.net/python-sort-multiple-keys/
- https://www.faqforge.com/linux/basics/how-to-find-duplicate-text-in-files-with-the-uniq-command-on-linux/
- https://www.askpython.com/python/dictionary/merge-dictionaries
