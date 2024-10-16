import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from stresnet.utils import AverageMeter, get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion,
        optimizer,
        scaler,
        device,
        save_dir,
    ):
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion, self.optimizer = criterion, optimizer
        self.scaler = scaler
        self.device = device
        self.save_dir = save_dir

        self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")

    def __to_numpy(self, x: torch.Tensor) -> np.ndarray:
        x_ = x.cpu().detach().numpy()
        return x_.reshape(1, -1)

    def _inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        x_ = self.__to_numpy(x)
        if self.scaler is None:  # not scaler
            return x_
        return self.scaler.inverse_transform(x_)

    def _inverse_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x_ = self._inverse_transform(x)
        y_ = self._inverse_transform(y)
        rmse = mean_squared_error(x_, y_, squared=False)
        return rmse

    def fit(self, model: nn.Module):
        for epoch in range(self.epochs):
            beg_time = time.time()
            model.train()
            losses = AverageMeter("train_loss")

            if epoch<60:  # if matrix size is too large to convergence, you can training task model in begining 40 epochs, therefore training R&E NN
                model.arranger.requires_grad_(False)
            else:
                model.arranger.requires_grad_(True)

            with tqdm(self.train_loader, dynamic_ncols=True, disable=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}]")

                for tr_data in pbar:
                    # tr_data: len=5, [32, 6, 32, 32]  [32, 2, 32, 32]  [32, 2, 32, 32]  [32, 28] GT:[32, 2, 32, 32]
                    # tr_X: close=[32, 6, 32, 32]  [32, 2, 32, 32]  [32, 2, 32, 32]  external:[32, 28]
                    tr_X = [d.to(self.device) for d in tr_data[:-1]]
                    tr_y = tr_data[-1].to(self.device)

                    self.optimizer.zero_grad()
                    out = model(*tr_X) * (tr_y!=0)  # filter missing position
                    # RMSE
                    loss = self.criterion(out, tr_y).sqrt()
                    loss.backward()
                    self.optimizer.step()

                    # inversed RMSE
                    rmse = self._inverse_loss(out, tr_y)
                    losses.update(rmse)

                    pbar.set_postfix(loss=losses.value)

            self.evaluate(model, epoch, beg_time)
        torch.save(model.arranger.state_dict(), 'arranger_taxi_STresnet.pth')

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None, beg_time=0) -> None:
        model.eval()
        losses = AverageMeter("valid_loss")
        NMAEs, ERs, RMSEs = AverageMeter("NMAE"), AverageMeter("ER"), AverageMeter("NMAE")

        for va_data in tqdm(self.valid_loader):
            va_X = [d.to(self.device) for d in va_data[:-1]]
            va_y = va_data[-1].to(self.device)

            out = model(*va_X) * (va_y!=0)  # filter missing position

            # inversed RMSE
            rmse = self._inverse_loss(out, va_y)
            losses.update(rmse)
            # calculate accuracy
            x, y = out, va_y
            x_ = torch.from_numpy(self._inverse_transform(x))
            y_ = torch.from_numpy(self._inverse_transform(y))

            NMAEs.update(NMAE(x_, y_))
            ERs.update(error_rate(x_, y_))
            RMSEs.update(RMSE(x_, y_))

        # print(f"loss: {losses.avg}")
        print(f"Testset[{epoch}]: avgER=\t{ERs.avg}\tavgNMAE=\t{NMAEs.avg}\tavgRMSE="
              f"\t{RMSEs.avg}\ttime=\t{time.time()-beg_time}")

        if epoch is not None:
            if losses.avg <= self.best_loss:
                self.best_acc = losses.avg
                torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))




def get_not_zero_position(inputs):
    return torch.clamp(torch.clamp(torch.abs(inputs), 0, 1e-32) * 1e36, 0, 1)

def NMAE(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = torch.abs(source - target)
    denominator = torch.abs(target)
    not_zero_pos = get_not_zero_position(target)
    return torch.sum(not_zero_pos * molecular) / torch.sum(denominator)

def RMSE(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    not_zero_pos = get_not_zero_position(target)

    return torch.sqrt(torch.pow(not_zero_pos*(source-target), 2).mean())

def error_rate(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = torch.pow(source - target, 2)
    denominator = torch.pow(target, 2)
    not_zero_pos = get_not_zero_position(target)
    return torch.pow(torch.sum(not_zero_pos * molecular) / torch.sum(denominator), 1 / 2)




