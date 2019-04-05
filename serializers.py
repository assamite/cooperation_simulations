import pickle

from numpy import ndarray, array


def get_serializers():
    return [get_array_ser, get_ndarray_ser]


def get_array_ser():
    return array, pickle.dumps, pickle.loads


def get_ndarray_ser():
    return ndarray, pickle.dumps, pickle.loads
