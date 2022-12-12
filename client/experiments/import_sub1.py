from os.path import dirname, join, realpath
from import_common import init_logging

a = 123

logger = init_logging(join(dirname(__file__), "import_sub1.log"))

def hello():
  # print(f'hello from import_sub1, id(logger): {id(logger)}')
  logger.info(f"Hello from import_sub1, a: {a}")