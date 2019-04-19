#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
import torch.utils.data as utils
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print(
        "WARNING: You are about to run on cpu, and this will likely run out of memory."
        + "\nYou can try setting batch_size=1 to reduce memory usage"
    )
    device = torch.device("cpu")


def create_grid(images, grid_size=6):
    index = np.random.choice(np.arange(len(images)), grid_size ** 2, replace=False)
    grid = np.array([]).reshape(0, images.shape[-1] * grid_size)
    for i in range(grid_size):
        line = np.concatenate(
            [images[index[j]] for j in range(i * grid_size, (i + 1) * grid_size)],
            axis=1,
        )
        grid = np.concatenate((grid, line), axis=0)
    return grid


def loadmat(f):
    return torch.Tensor(pd.read_csv(f, sep=" ").values).view(-1, 1, 28, 28)


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.sf = scale_factor

    def forward(self, x):
        return self.interp(x, scale_factor=self.sf)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(64, 256, 5),
            nn.ELU(),
        ).to(device)
        self.mu = nn.Linear(256, 100).to(device)
        self.logvar = nn.Linear(256, 100).to(device)
        self.generated = nn.Linear(100, 256).to(device)
        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(256, 64, 5, padding=4),
            nn.ELU(),
            Interpolate(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            Interpolate(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, 3, padding=2),
            nn.Sigmoid(),
        ).to(device)

    def encode(self, x):
        out = self.encoder(x).squeeze()
        return self.mu(out), self.logvar(out)

    def reparam(self, mu, logvar):
        seed = torch.Tensor(np.random.normal(0, 1, 100)).to(device)
        if self.training:
            return seed.mul(logvar.exp().pow(0.5)).add(mu)
        else:
            return mu

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        seed = self.reparam(mu, logvar)
        gen = self.generated(seed)
        gen = gen.view(-1, 256, 1, 1)
        return self.decode(gen), mu, logvar


def dkl(mu, logvar):
    DKL = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
    DKL /= len(logvar) * 784
    return DKL


def bce(out, X):
    BCE = nn.functional.binary_cross_entropy(
        out.view(-1, 784), X.view(-1, 784), size_average=False
    )
    return BCE


def elbo(X, out, mu, logvar):
    DKL = dkl(mu, logvar)
    BCE = nn.BCELoss(reduction="sum")(out, X)
    return BCE + DKL


if __name__ == "__main__":
    batch_size = 16
    EPOCHS = 40
    train = loadmat("binarized_mnist_train.amat")
    train_dataset = utils.TensorDataset(train)
    trainloader = utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    valid = loadmat("binarized_mnist_valid.amat")
    valid_dataset = utils.TensorDataset(valid)
    validloader = utils.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test = loadmat("binarized_mnist_test.amat")
    test_dataset = utils.TensorDataset(test)
    testloader = utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = VAE()
    optimizer = Adam(model.parameters(), lr=3e-4)
    for e in range(EPOCHS):
        ELBOs = []
        for X in trainloader:
            optimizer.zero_grad()
            X = X[0].to(device)
            out, mu, logvar = model.forward(X)
            ELBO = elbo(X, out, mu, logvar)
            ELBO.backward()
            optimizer.step()
            ELBOs.append(float(ELBO))

        vELBOs = []
        for svalid in validloader:
            svalid = svalid[0].to(device)
            vout, vmu, vlogvar = model.forward(svalid)
            vELBOs.append(float(elbo(svalid, vout, vmu, vlogvar)))

        print(
            f"Epoch {e}: train_loss: {-np.mean(ELBOs):.5f}, valid_loss: {-np.mean(vELBOs):.5f}"
        )

    generated = np.array([]).reshape(0, 28, 28)
    for batch in testloader:
        out, mu, logvar = model.forward(batch[0].to(device))
        generated = np.concatenate(
            (generated, out.view(-1, 28, 28).detach().cpu().numpy()), axis=0
        )

    plt.matshow(create_grid(generated))
    plt.show()
