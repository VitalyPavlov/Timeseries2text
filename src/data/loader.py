import torch
import pandas as pd
from pydoc import locate
from pathlib import Path
from torch.utils.data import WeightedRandomSampler


def create_loader(
    cfg: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    sampler: WeightedRandomSampler,
):
    """Initialization of Pytorch Dataloader."""

    # Dataset for train
    train_dataset = locate(cfg.dataset.loader_train)(
        data=train_df,
        augmentation=locate(cfg.dataset.augmentation),
        preprocessing=locate(cfg.dataset.preprocessing),
    )

    # Dataset for validation
    valid_dataset = locate(cfg.dataset.loader_valid)(
        data=valid_df,
        preprocessing=locate(cfg.dataset.preprocessing),
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=my_collate,
            shuffle=False,
            num_workers=min(cfg.train.num_workers, cfg.train.batch_size_train),
            batch_size=cfg.train.batch_size_train,
        ),
        "valid": torch.utils.data.DataLoader(
            valid_dataset,
            collate_fn=my_collate,
            shuffle=False, # False
            num_workers=min(cfg.train.num_workers, cfg.train.batch_size_valid),
            batch_size=cfg.train.batch_size_valid,
        ),
    }

    return dataloaders


#############################
### Helper for dataloader ###
#############################

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]
