import sys
import logging
from datetime import datetime
# from time import gmtime, strftime, time


def creat_logger(name, silent=False, to_disk=False, log_file=None, prefix=None):
    """ simplify from san_mrc """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s [%(name)-6s]-[%(levelname)-4s] %(message)s',
    #                               datefmt='%Y-%m-%d %H:%M:%S',
    #                               style='%')
    formatter = logging.Formatter('%(asctime)s [%(name)s]-[%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S',
                                  style='%')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if to_disk:
        prefix = prefix if prefix is not None else 'log'
        log_file = log_file if log_file is not None else "{}.{}.log".format(prefix, datetime.now().strftime("%b-%d_%H-%M-%S"))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # disable elmo info
    logger.propagate = False
    return logger