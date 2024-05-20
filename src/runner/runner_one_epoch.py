import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Any
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunnerTrain:
    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: int,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict,
        scaler: GradScaler,
        device: torch.device,
        writer: SummaryWriter,
        save_pred: bool,
        batch_size: int
    ) -> None:
        self.run_count: int = 0
        self.model = model
        self.vocab_size = vocab_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scaler = scaler
        self.device = device
        self.writer = writer
        self.save_pred = save_pred
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        self.loss_val: float = 0
        self.batch_size = batch_size

    def run_one_epoch(self) -> None:
        self.model.train()
        epoch_loss_train = AverageMeter()

        for inputs, labels in tqdm(
            self.dataloaders["train"], desc="Train", ncols=80, leave=False
        ):
            if self.batch_size > 1:
                inputs, labels = add_padding(inputs, labels)

            inputs = torch.cat(inputs).to(self.device)
            # print(torch.cat(labels).shape)
            # labels = torch.cat(labels).permute(1,0,2).to(self.device)
            labels = torch.cat(labels).to(self.device)

            self.optimizer.zero_grad()
            if self.scaler:
                with autocast():
                    outputs = self.model(inputs, labels)
                    loss = self.loss_fn(outputs.reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1))

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs, labels)
                    loss = self.loss_fn(outputs.reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1))

                    loss.backward()
                    self.optimizer.step()

            epoch_loss_train.update(loss.item(), inputs.size(0))

            if self.save_pred:
                self.y_true_batches += [labels.data().cpu().numpy()]
                self.y_pred_batches += [outputs.data().cpu().numpy()]

            self.writer.add_scalar(
                "BatchLoss/train", epoch_loss_train.val, self.run_count
            )
            self.run_count += 1

        self.loss_val = epoch_loss_train.avg


class RunnerValid:
    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: int,
        teacher_forcing_ratio: int,
        loss_fn: torch.nn.modules.loss._Loss,
        metric_fn: Any,
        dataloaders: dict,
        device: torch.device,
        writer: SummaryWriter,
        save_pred: bool,
        batch_size: int
    ) -> None:
        self.run_count: int = 0
        self.model = model
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.dataloaders = dataloaders
        self.device = device
        self.writer = writer
        self.save_pred = save_pred
        self.fig: plt = None
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        self.loss_val: float = 0
        self.metric_val: float = 0
        self.metric_name: str = "Metric"
        self.batch_size = batch_size

    def run_one_epoch(self, plot_pred: bool):
        self.model.eval()
        epoch_loss_valid = AverageMeter()
        epoch_metric_valid = AverageMeter()

        for inputs, labels in tqdm(
            self.dataloaders["valid"], desc="Eval", ncols=80, leave=False
        ):
            if self.batch_size > 1:
                inputs, labels = add_padding(inputs, labels)

            inputs = torch.cat(inputs).to(self.device)
            labels = torch.cat(labels).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(inputs, labels, self.teacher_forcing_ratio)
                # print(outputs.shape, labels.shape)
                loss = self.loss_fn(outputs[:,1:,:].reshape(-1, self.vocab_size), labels[:,1:].reshape(-1))

                # outputs = torch.sigmoid(outputs)
                # outputs = (outputs > 0.5).long()

                y_true = labels[:,1:].reshape(-1).data.cpu().numpy()
                y_pred = outputs[:,1:,:].reshape(-1, self.vocab_size).argmax(1).data.cpu().numpy()
                # name, metric = self.metric_fn(y_pred, y_true)
                # print(y_true.shape, y_pred.shape)
            
            epoch_loss_valid.update(loss.item(), inputs.size(0))
            # epoch_metric_valid.update(metric, inputs.size(0))

            if self.save_pred:
                self.y_true_batches += list(np.ravel(y_true))
                self.y_pred_batches += list(np.ravel(y_pred))

            self.writer.add_scalar(
                "BatchLoss/valid", epoch_loss_valid.val, self.run_count
            )
            self.run_count += 1

        name, metric = self.metric_fn(self.y_pred_batches, self.y_true_batches)
        epoch_metric_valid.update(metric, inputs.size(0))

        self.loss_val = epoch_loss_valid.avg
        self.metric_name = name
        self.metric_val = epoch_metric_valid.avg



##############
### Helper ###
##############

def add_padding(inputs, labels):
    max_len = max([x.shape[2] for x in inputs])
    for i in range(len(inputs)):
        padding = torch.zeros((1,32,max_len-inputs[i].shape[2]))
        inputs[i] = torch.cat((inputs[i], padding), 2)

    max_len = max([x.shape[1] for x in labels])
    for i in range(len(labels)):
        padding = torch.Tensor([9]).repeat((1, max_len-labels[i].shape[1])).long()
        labels[i] = torch.cat((labels[i], padding), 1)

    return inputs, labels
