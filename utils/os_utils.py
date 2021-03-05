import argparse


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


