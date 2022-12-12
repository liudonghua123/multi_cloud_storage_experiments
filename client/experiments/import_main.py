from import_sub1 import hello, a

from os.path import dirname, join, realpath
from import_common import init_logging

logger = init_logging(join(dirname(__file__), "import_main.log"))

# TODO: change a will not change the a value in import_sub1
a = 456
# Need to use importlib.a to modify the value
import import_sub1
import_sub1.a = 789

hello()

# print(f'import_main, id(logger): {id(logger)}')
logger.info(f"import_main a: {a}")
