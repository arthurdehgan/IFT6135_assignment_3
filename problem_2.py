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
    """
    Creates a grid array from a list of squared images.

    Parameters
    ----------
    images: a list or numpy array images of shape (n_images, width, height)
        the images to put on grid form.
    grid_size: int, (default 6)
        the size of the grid of images that will be returned.

    Returns
    -------
    grid: numpy array, of shape (image_size * grid_size, image_size * grid_size)
        the concatenated images in a grid
    """
    width = images[0].shape[0]
    n_images = len(images)
    index = np.random.choice(np.arange(n_images), grid_size ** 2, replace=False)
    grid = np.array([]).reshape(0, width * grid_size)
    for i in range(grid_size):
        line = np.concatenate(
            [images[index[j]] for j in range(i * grid_size, (i + 1) * grid_size)],
            axis=1,
        )
        grid = np.concatenate((grid, line), axis=0)
    return grid


def loadmat(f):
    """
    Loads a binary MNIST dataset.

    Parameters
    ----------
    f: string
        the file path.
    """
    return torch.Tensor(pd.read_csv(f, sep=" ").values).view(-1, 1, 28, 28)


def normal_sample(mu, logvar, size):
    """
    Generates a tensor according to the using the normal law

    Parameters
    ----------
    mu: torch.Tensor
        the means of each sample of the final tensor
    logvar: torch.Tensor:
        the log of the variance of each sample of the final tensor
    size: int
        the size of the tensor to generate

    Returns
    -------
    torch.Tensor
        the generated samples
    """
    seed = torch.Tensor(np.random.normal(0, 1, size))
    return seed.mul((0.5 * logvar).exp()).add(mu)


class Interpolate(nn.Module):
    """
    Torch module to perform 2D upscaling via interpolation.

    Parameters
    ----------
    scale_factor: int
        the factor that will be used to upscale.
    """

    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.sf = scale_factor

    def forward(self, x):
        """
        forward pass of the module.

        Parameters
        ----------
        X: torch.Tensor
            The data.

        Returns
        -------
        Z: torch.Tensor
            The upscaled data by a factor of scale_factor.
        """
        return self.interp(x, scale_factor=self.sf)


class VAE(nn.Module):
    """
    Torch module, Variational AutoEncoder.


    Parameters
    ----------
    latent_size: int (default 100)
        the size of the latent space of the autoencoder

    Attributes
    ----------
    encoder: nn.Sequential
        the encoder.
    mu: nn.Linear
        the means of the learned distribution.
    logvar: nn.Linear
        the log of the variance of the learned distribution.
    sample: nn.Linear
        the linear layer used for the genration of images from the latent space. # TODO
    decoder: nn.Sequential
        the decoder.
    """

    def __init__(self, latent_size=100):
        super(VAE, self).__init__()

        self.latent_size = latent_size
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
        self.mu = nn.Linear(256, latent_size).to(device)
        self.logvar = nn.Linear(256, latent_size).to(device)
        self.sample = nn.Linear(latent_size, 256).to(device)
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
        """
        Encodes the data in the latent space

        Parameters
        ----------
        x: torch.Tensor
            the data of shape (batch_size, 1, width, height)

        Returns
        -------
        mu: torch.Tensor
            the means of the learned distribution.
        logvar: torch.Tensor
            the log of the variance of the learned distribution.
        """
        out = self.encoder(x).squeeze()
        return self.mu(out), self.logvar(out)

    def reparam(self, mu, logvar):
        """
        Reparametrization trick, without it backprop is not possible.

        Parameters
        ----------
        mu: torch.Tensor
            the means of the learned distribution.
        logvar: torch.Tensor
            the log of the variance of the learned distribution.

        Returns
        -------
        A generated vector according to the unit normal law during training, the mean vector of the
            distribution during testing
        """
        if self.training:
            normal_sample(mu, logvar, self.latent_size).to(device)
        else:
            return mu

    def decode(self, z):
        """
        decodes the generated vector back into images.

        Parameters
        ----------
        x: torch.Tensor
            the stocasticaly generated vector of size latent_size.

        Returns
        -------
        gen: torch.Tensor
            a tensor of reconstructed images of shape (batch_size, 1, width, height)
        """
        z = self.sample(z)
        z = z.view(-1, 256, 1, 1)
        return self.decoder(out)

    def forward(self, x):
        """
        Runs the forward pass of the VAE.

        Parameters
        ----------
        x: torch.Tensor
            the data of shape (batch_size, 1, width, height)

        Returns
        -------
        gen: torch.Tensor
            a tensor of reconstructed images of shape (batch_size, 1, width, height)
        mu: torch.Tensor
            the means of the learned distribution.
        logvar: torch.Tensor
            the log of the variance of the learned distribution.
        """
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        gen = self.decode(z)
        return gen, mu, logvar


def dkl(mu, logvar):
    """
    Computes the KL divergence.

    Parameters
    ----------
    mu: torch.Tensor
        a tensor of means of distributions
    logvar: torch.Tensor
        a tensor of log of variances of distributions

    Returns
    -------
    DKL:
        the KL divergence
    """
    DKL = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
    DKL /= len(mu) * 784
    return DKL


def elbo(X, out, mu, logvar):
    """
    Copmputes the Evidence Lower BOund.

    Parameters
    ----------
    X: torch.Tensor
        the original data
    out: torch.Tensor
        the predicted values
    mu: torch.Tensor
        the tensor of means of distributions computed by the VAE when encoding
    logvar: torch.Tensor
        the tensor of log of variances of distributions computed by the VAE when encoding

    Returns
    -------
    ELBO: torch.Tensor
        the negative ELBO (which is positive, because we will minimize this value instead of
        maximizing the ELBO)
    """
    DKL = dkl(mu, logvar)
    BCE = nn.BCELoss(reduction="sum")(out, X)
    # return BCE - DKL
    return BCE + DKL


def probability_density_function(z, mu, logvar):
    """
    Computes the log the probabily density function of z, following a multivariate gaussian.

    Parameters
    ----------
    z: torch.Tensor
        a 3D tensor of shape (batch_size, n_imp, latent_size) with:
            n_imp: the number of importance samples
            latent_size : the size of the latent space of the autoencoder
    mu: torch.Tensor
        a tensor of means of multivariate distributions of shape (batch_size, latent_size)
    logvar: torch.Tensor
        a tensor of log of variances of multivariate distributions
        of shape (batch_size, latent_size)

    Returns
    -------
    torch.Tensor
        the log the probabily density function of z, following a multivariate gaussian,
        of shape (batch_size, n_imp)
    """
    latent_size = mu.shape[1]
    mu = mu.view(-1, 1, latent_size)
    logvar = logvar.view(-1, 1, latent_size)
    log_det_cov = logvar.sum(dim=2)  # log(det(Sigma)) with Sigma the covariance matrix
    inv_sigma = 1. / logvar.exp()
    # each row is the diagonal entries of the inverse of covariance matrix
    # the inverse of a diagonal matrix is the inverse of the elements
    log_exp = (inv_sigma * (z - mu) ** 2).sum(dim=2)
    return (-k / 2) * np.log(2 * np.pi) - .5 * log_det_cov - .5 * log_exp


def log_likelihood(model, X, Z):
    """
    Computes the log likelihood of Z from a model of X.

    Parameters
    ----------
    model: torch.nn.Module
        a VAE model.
    X: torch.Tensor
        the data, of shape (batch_size, width*height)
    Z: torch.Tensor
        a 3D tensor of shape (batch_size, n_samples, latent_size) with:
            n_samples: the number of importance samples (200-300)
            latent_size : the size of the latent space of the autoencoder (100)

    Returns
    -------
    log(p(x)): # TODO
        the log likelihood
    """
    batch_size = X.shape[0]
    n_samples = Z.shape[1]

    log_p_xz = torch.Tensor(batch_size, n_samples)
    mu, logvar = model.encode(X.view(batch_size, 1, 28, 28))

    for i in range(n_samples):
        out = model.decode(Z[:, i, :])

        # reconstruction error using BCE
        log_p_xz[:, i] = -nn.functional.binary_cross_entropy(
            out.view(-1, 784), X.view(-1, 784), reduction="none"
        ).sum(dim=1)
    # q(z|x) follows a multivariate normal distribution of mu, sigma^2
    log_q_zx = probability_density_function(Z, mu, logvar)

    # p(z) follows a standard multivariate normal distribution
    log_p_z = probability_density_function(
        Z, torch.zeros_like(mu), torch.zeros_like(logvar)
    )

    log_p_x = log_p_xz + log_p_z - log_q_zx
    return log_p_x.logsumexp(dim=1) - np.log(n_samples)


if __name__ == "__main__":
    mode = "elbo"
    batch_size = 4
    EPOCHS = 20
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
        losses = []
        model.train()
        for X in trainloader:
            optimizer.zero_grad()
            X = X[0].to(device)
            out, mu, logvar = model.forward(X)
            if mode == "elbo":
                loss = elbo(X, out, mu, logvar)
            else:
                for _ in range(K)
                    normal_sample(mu, logvar, )
                Z = torch.Tensor()  # TODO
                loss = log_likelihood(model, X, Z)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))

        vlosses = []
        model.eval()
        for svalid in validloader:
            svalid = svalid[0].to(device)
            vout, vmu, vlogvar = model.forward(svalid)
            if mode == "elbo":
                vloss = float(elbo(svalid, vout, vmu, vlogvar))
            else:
                Z = torch.Tensor()  # TODO
                vloss = float(log_likelihood(model, X, Z).mean())
            vlosses.append(vloss)

        print(
            f"Epoch {e}: train_loss: {-np.mean(losses):.5f}, valid_loss: {-np.mean(vlosses):.5f}"
        )

    generated = np.array([]).reshape(0, 28, 28)
    for batch in testloader:
        out, mu, logvar = model.forward(batch[0].to(device))
        generated = np.concatenate(
            (generated, out.view(-1, 28, 28).detach().cpu().numpy()), axis=0
        )

    plt.matshow(create_grid(-generated), cmap=plt.cm.gray_r)
    plt.savefig("generated_grid.png")
