#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam


def loadmat(f):
    return torch.Tensor(pd.read_csv(f, sep=" ").values).view(-1, 1, 28, 28)


def elbo(reconstruction, x, mu, logvar):
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
        )
        self.mu = nn.Linear(256, 100)
        self.sig = nn.Linear(256, 100)
        self.decoder = nn.Sequential(
            nn.Linear(100, 256),
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
        )

    def encode(self, x):
        out = self.encoder(x)
        print(out.shape)
        return self.mu(out), self.sig(out)

    def decode(self, mu, sig):
        seed = np.random.normal(0, 1, 100)
        if self.training():
            gen = seed * sig + mu
        else:
            gen = mu
        return self.decoder(gen)

    def forward(self, x):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    train = loadmat("binarized_mnist_train.amat")
    model = VAE()
    model.forward(train[:10])
    optimizer = Adam(model.parameters(), lr=3e-4)
    for _ in 20:
        out = model.forward(dat)
        model.backward(elbo(out))
        optimizer.step()
