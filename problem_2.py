#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam

if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print(
        "WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage"
    )
    device = torch.device("cpu")


def loadmat(f):
    return torch.Tensor(pd.read_csv(f, sep=" ").values).view(-1, 1, 28, 28)


def elbo(input_size, reconstruction, x, mu, logvar):
    """ELBO assuming entries of x are binary variables, with closed form DKL."""
    bce = torch.nn.functional.binary_cross_entropy(
        reconstruction, x.view(-1, input_size)
    )
    DKL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    DKL /= x.view(-1, input_size).data.shape[0] * input_size
    return bce + DKL


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
        self.sig = nn.Linear(256, 100).to(device)
        self.generated = nn.Linear(100, 256).to(device)
        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(256, 64, 5, padding=4),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, 3, padding=2),
        ).to(device)

    def encode(self, x):
        out = self.encoder(x).squeeze()
        return self.mu(out), self.sig(out)

    def reparam(self, mu, sig):
        seed = torch.Tensor(np.random.normal(0, 1, 100))
        if self.training:
            gen = seed * sig + mu
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
    for _ in EPOCHS:
        for i in range(0, len(train), batch_size):
            optimizer.zero_grad()
            X = train[:batch_size].to(device)
            out, mu, logvar = model.forward(X)
            elbo(28, out, X, mu, logvar)
            loss = criterion(out, X)
            loss.backward()
            optimizer.step()
