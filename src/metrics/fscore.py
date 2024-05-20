import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def get_metrics(pred, true, average='micro'):
    name = "F1_score"

    p = np.array(pred).reshape(-1, 1)
    t = np.array(true).reshape(-1, 1)

    f_score = f1_score(t, p, average=average)

    return name, f_score
