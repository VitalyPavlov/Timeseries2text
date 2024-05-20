import torch
import shutil
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from fire import Fire
from addict import Dict
from pydoc import locate
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

from src.utils.read_config import model_cfg
from src.utils.set_seed import seed_everything
from src.data.loader import create_loader
from src.runner.runner_one_epoch import RunnerTrain, RunnerValid
from src.models.transformer import NoamOpt


@hydra.main(version_base=None, config_path='./config', config_name="default")
def main(cfg: DictConfig):
    # create logger
    logger = logging.getLogger(__name__)
    log_dir = Path(cfg.path.logger).joinpath(cfg.info.exp_name)
    fh = logging.FileHandler(log_dir.joinpath("run.log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    # set random seed for all modules
    seed_everything(seed=cfg.train.seed)

    # read data
    file_name = Path(cfg.path.base).joinpath(cfg.path.train_file)
    
    logger.info(f"file name: {file_name}")
    data = pd.read_parquet(file_name)

    if cfg.dataset.limit_by_length:
        # data = data.iloc[:500]
        data['len_target'] = data['target'].apply(lambda x: len(x))
        data = data[data.len_target <= cfg.dataset.limit_by_length].reset_index(drop=True)

    stratification = {k:v for k,v in zip(data.stratification.unique(), range(len(data.stratification.unique())))}
    data['stratification'] = data['stratification'].map(stratification)

    train_df = data[data.fold == 'train'].reset_index(drop=True)
    valid_df = data[data.fold == 'test'].reset_index(drop=True)

    class_sample_count = np.array(
        [
            len(np.where(train_df.stratification == t)[0])
            for t in np.unique(data.stratification)
        ]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t-1] for t in train_df.stratification])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weight, num_samples=len(samples_weight), replacement=True
    )

    # datasets and dataloader
    dataloaders = create_loader(cfg, train_df, valid_df, sampler)

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"fold: {cfg.dataset.fold}")
    logger.info(f"device {device}")
    logger.info(
        f"train_size: {len(train_df)}, valid_size: {len(valid_df)}, sampler: {len(samples_weight)}"
    )
    logger.info("epoch, lr, train_loss, valid_loss, valid_metric, best_metric")

    if cfg.train.model == 'src.models.seq2seq.Seq2Seq':
        vocab_size = cfg.rnn.vocab_size
        model = locate(cfg.train.model)(cfg.rnn)

    elif cfg.train.model == 'src.models.transformer.GPT':
        vocab_size = cfg.transformer.vocab_size
        model = locate(cfg.train.model)(cfg.transformer)

    model = model.to(device)

    # add pretained weights
    if cfg.path.pretrained_weights:
        pretrained_weights = cfg.path.pretrained_file
        model.load_state_dict(torch.load(pretrained_weights))

    if not cfg.info.debug_mode:
        writer = SummaryWriter(log_dir=log_dir)

    # Parameters for training
    scaler = GradScaler() if cfg.train.fp16 else None
    loss_fn = locate(cfg.train.loss)()
    metric_fn = locate(cfg.train.metric)
    if cfg.train.model == 'src.models.transformer.GPT':
        optimizer = model.configure_optimizers(float(cfg.transformer.weigth_decay), float(cfg.train.lr), (float(cfg.transformer.beta1), float(cfg.transformer.beta2)))
    else:
        optimizer = locate(cfg.train.optimizer)(model.parameters(), lr=float(cfg.train.lr))

    # if cfg.train.model == 'src.models.transformer.GPT':
    #     # model_size = sum(p.numel() for p in model.encoder.parameters()) // 20
    #     # scheduler = NoamOpt(model_size, cfg.transformer.warmup, optimizer)
    #     # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.002, steps_per_epoch=1, epochs=cfg.train.epochs, div_factor=2)
    #     scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.8, total_iters=cfg.train.epochs)
    # else:
    scheduler = locate(cfg.train.scheduler)(
        optimizer,
        mode="max",
        factor=float(cfg.train.reduce_lr_factor),
        patience=int(cfg.train.reduce_lr_patience),
        min_lr=float(cfg.train.reduce_lr_min),
        verbose=True,
    )

    train_runner = RunnerTrain(
        model,
        vocab_size=vocab_size,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloaders=dataloaders,
        scaler=scaler,
        device=device,
        writer=writer,
        save_pred=False,
        batch_size=cfg.train.batch_size_train,
    )

    valid_runner = RunnerValid(
        model,
        vocab_size=vocab_size,
        teacher_forcing_ratio=cfg.train.teacher_forcing_ratio,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        dataloaders=dataloaders,
        device=device,
        writer=writer,
        save_pred=True,
        batch_size=cfg.train.batch_size_valid,
    )

    start_time = time.time()
    best_metrics = 0.0
    early_stoping = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(cfg.train.epochs):
        lr = optimizer.param_groups[0]["lr"]

        plot_pred = False # True if epoch % 2 == 0 else False

        train_runner.run_one_epoch()
        valid_runner.run_one_epoch(plot_pred)
        
        # if cfg.train.model == 'src.models.transformer.GPT':
        #     # print(lr)
        #     # optimizer.step()
        #     scheduler.step()
        # else:
        scheduler.step(valid_runner.metric_val)

        if valid_runner.metric_val > best_metrics:
            best_metrics = valid_runner.metric_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, cfg.path.weights)
            early_stoping = 0
            model_updated = "*"
        else:
            early_stoping += 1
            model_updated = ""

        logger.info(
            f"{epoch}, {lr:6f}, {train_runner.loss_val:.3f}, {valid_runner.loss_val:.3f}, {valid_runner.metric_val:.3f}, {best_metrics:.3f}{model_updated}"
        )

        metric_name = valid_runner.metric_name
        writer.add_scalar("Loss/train", train_runner.loss_val, epoch)
        writer.add_scalar("Loss/valid", valid_runner.loss_val, epoch)
        writer.add_scalar(f"{metric_name}/valid", valid_runner.metric_val, epoch)
        writer.add_scalar("Learning rate", lr, epoch)

        if early_stoping > cfg.train.early_stop_patience:
            break

    writer.add_hparams(
        {"lr": cfg.train.lr, "bsize": cfg.train.batch_size_train},
        {"hparam/metric": valid_runner.metric_val},
        run_name="summary",
    )
    writer.flush()

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print("Best val metrics: {:4f}".format(best_metrics))


if __name__ == "__main__":
    main()
