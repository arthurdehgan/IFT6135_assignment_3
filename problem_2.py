#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print(
        "WARNING: You are about to run on cpu, and this will likely run out of memory."
        + "\nYou can try setting batch_size=1 to reduce memory usage"
    )
    device = torch.device("cpu")


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
        seed = torch.Tensor(np.random.normal(0, 1, 100))
        if self.training:
            gen = seed.mul(logvar.exp().pow(0.5)).add(mu)
        else:
            gen = mu
        return gen

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        seed = self.reparam(mu, logvar)
        gen = self.generated(seed)
        gen = gen.view(-1, 256, 1, 1)
        return self.decode(gen), mu, logvar


if __name__ == "__main__":
    train = loadmat("binarized_mnist_train.amat")
    model = VAE()
    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCELoss()
    batch_size = 128
    EPOCHS = 20
    for e in range(EPOCHS):
        print(f"Epoch: {e}")
        for i in range(0, len(train), batch_size):
            optimizer.zero_grad()
            X = train[:batch_size].to(device)
            out, mu, logvar = model.forward(X)
            out = out.view(-1, 1, 28, 28)
            DKL = torch.sum(-1 - logvar + mu.pow(2) + logvar.exp()).mul(0.5)
            reconstruction = criterion(out, X)
            ELBO = reconstruction - DKL
            if i % 1024 == 0:
                # print(f"    Iteration: {i}, loss: {float(ELBO):.5f}", end="\r")
                print(f"    Iteration: {i}, loss: {float(ELBO):.5f}")
            ELBO.backward()
            optimizer.step()
