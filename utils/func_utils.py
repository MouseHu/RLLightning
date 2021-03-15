import argparse
from itertools import chain
from numbers import Number
import numpy as np


def str2bool(value):
    value = str(value)
    if isinstance(value, bool):
        return value
    if value.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Get ' + str(value.lower()))


def remove_color(key):
    for i in range(len(key)):
        if key[i] == '@':
            return key[:i]
    return key


def array_min2d(x):
    x = np.array(x).astype(np.float32)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


# merge all dicts and return average from each value
# ignore those can't be averaged
def merge_dicts(infos):
    keys = list(set(list(chain.from_iterable([list(info.keys()) for info in infos]))))
    merged_info = dict()
    for k in keys:
        values = [np.array(info.get(k, np.nan)) for info in infos]
        values = np.array(values).reshape(-1)
        if all([isinstance(v, Number) for v in values]):  # only scalars are recorded
            mean_value = np.nanmean(values)
            merged_info[k] = mean_value
    return merged_info

