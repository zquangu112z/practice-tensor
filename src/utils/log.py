import logging
import coloredlogs
from pprint import pformat

coloredlogs.install(
    level='INFO',
    fmt='%(asctime)s %(hostname)s %(filename)15s[%(lineno)d] %(levelname)s - %(message)s',
)

# log = logging.warning


def log(*args):
    # logging.warning(pformat(args))
    logging.warning(args)
