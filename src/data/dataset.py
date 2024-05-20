import torch
import numpy as np
import pandas as pd
from typing import Optional


# classes for data loading and preprocessing
class Dataset_Train:
    def __init__(
        self,
        data: pd.DataFrame,
        augmentation: Optional[str] = None,
        preprocessing: Optional[str] = None,
    ):
        self.data = data
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        trace = np.array(self.data.loc[i, "seq"])
        target = np.array([int(x) for x in self.data.loc[i, "target"]])
        
        # apply preprocessing
        if self.preprocessing:
            trace = self.preprocessing(trace)
            
        # apply augmentations
        if self.augmentation:
            trace, target = self.augmentation(trace, target)

        # encoding
        # label = np.zeros((len(target) + 2, 10))
        # for j in range(len(target)):
        #     label[j + 1][target[j]] = 1
            
        #SOS and EOS
        # label[0][:] = [1] * label.shape[1]
        # label[-1][0] = 1
        label = np.concatenate([np.array([1]), target, np.array([0])])

        trace = np.expand_dims(trace, axis=0)
        label = np.expand_dims(label, axis=0)

        return torch.Tensor(trace), torch.Tensor(label).long()

    def __len__(self):
        return len(self.data)
