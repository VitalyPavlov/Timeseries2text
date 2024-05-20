import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter


def inverse_values(trace: np.array, label: np.array, p: int = 0.3):
    if random.uniform(0, 1) < p:
        return trace[:,::-1].copy(), label[::-1].copy()
    return trace, label


def add_values(trace: np.array, label: np.array, p: int = 0.5):
    if len(label) <= 6 and random.uniform(0, 1) < p:
        new_trace = trace[:, ::-1].copy()
        new_label = label[::-1].copy()
        label[-1] *= 2
        
        return np.concatenate((trace, new_trace), axis=1), np.concatenate((label, new_label[1:]), axis=0)
    return trace, label

def blur(trace: np.array, p: int = 0.5):
    if random.uniform(0, 1) < p:
        sigma = random.uniform(0.5, 4)
        trace = gaussian_filter(trace, sigma=sigma)
        return trace.copy()
    return trace

def aug_func(trace, label):
    trace, label = inverse_values(trace, label)
    trace = blur(trace)

    return trace, label
