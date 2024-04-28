import argparse
import copy
import os
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, h5_file: str):
        super(BaseDataset, self).__init__()
        self.f = h5py.File(h5_file, "r")

    def __len__(self) -> int:
        return len(self.f["lr"])


class TrainDataset(BaseDataset):
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.expand_dims(self.f["lr"][idx] / 255.0, 0), np.expand_dims(
            self.f["hr"][idx] / 255.0, 0
        )


class EvalDataset(BaseDataset):
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.expand_dims(self.f["lr"][str(idx)][:, :] / 255.0, 0), np.expand_dims(
            self.f["hr"][str(idx)][:, :] / 255.0, 0
        )


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class AverageMeter(object):
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


def calc_psnr(img1, img2):
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))


def main(args):
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters()},
            {"params": model.conv3.parameters(), "lr": args.lr * 0.1},
        ],
        lr=args.lr,
    )
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.num_epochs
        )
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(
            total=(len(train_dataset) - len(train_dataset) % args.batch_size)
        ) as t:
            t.set_description("epoch: {}/{}".format(epoch + 1, args.num_epochs))
            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.scheduler:
                    scheduler.step()
                t.set_postfix(loss="{:.6f}".format(epoch_losses.avg))
                t.update(len(inputs))
        torch.save(
            model.state_dict(),
            os.path.join(args.outputs_dir, "epoch_{}.pth".format(epoch)),
        )
        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        print("eval psnr: {:.2f}".format(epoch_psnr.avg))
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
    print("best epoch: {}, psnr: {:.2f}".format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, "best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scheduler", action="store_true")
    args = parser.parse_args()
    args.outputs_dir = os.path.join(args.outputs_dir, "x{}".format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    main(args)
