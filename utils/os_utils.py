import argparse
import numpy as np
from numbers import Number
from itertools import chain

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


# merge all dicts and return average from each value
# ignore those can't be averaged
def merge_dicts(infos):
    keys = list(set(list(chain.from_iterable([list(info.keys()) for info in infos]))))
    merged_info = dict()
    for k in keys:
        values = [info.get(k, np.nan) for info in infos]
        if all([isinstance(v, Number) for v in values]):  # only scalars are recorded
            mean_value = np.nanmean(values)
            merged_info[k] = mean_value
    return merged_info
