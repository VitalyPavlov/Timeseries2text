import numpy as np


def normalization_by_max(trace: np.array):
    return trace / max(trace)

def reshaping(trace: np.array, shape_dim: int):
    return trace.reshape(shape_dim, -1)

def preprocessing_32(trace: np.array):
    trace = normalization_by_max(trace)
    trace = reshaping(trace, 32)
    return trace.copy()

def preprocessing_100(trace: np.array):
    trace = normalization_by_max(trace)
    trace = reshaping(trace, 100)
    return trace[:-4].reshape(3,32,-1).mean(axis=0)
