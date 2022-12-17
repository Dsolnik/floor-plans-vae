import pickle
import pandas as pd
import os
import time

import numpy as np
import torch
from pyro.contrib.examples.util import MNIST
import torch.nn as nn
import torchvision.transforms as transforms
from torch import Tensor
from math import prod
from typing import Tuple

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

X = 64
Y = 64

NUM_PIXELS = X * Y

PICKLE_DIR = "/home/solnik/floor_plans/Final Projects/pickles"


def save(obj, name, pickle_dir=PICKLE_DIR):
    print(f"SAVING {pickle_dir}/{name} ")
    with open(f"{pickle_dir}/{name}", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(name, pickle_dir=PICKLE_DIR):
    with open(f"{pickle_dir}/{name}", "rb") as handle:
        return pickle.load(handle)


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dtype=None):
        super().__init__(
            nn.Linear(in_dim, hidden_dim, dtype=dtype),
            nn.Softplus(),
            nn.Linear(hidden_dim, out_dim),
        )


class LocScale(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.loc = nn.Linear(in_dim, out_dim)
        self.scale = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Softplus(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.loc(x), self.scale(x)


class Binary(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
        )


class AmortizedVAE(nn.Module):
    """
    Standard variational encoder model
    for binary image data with a
    Gaussian latent variable.
    """

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int, use_cuda: bool):
        super().__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Flatten(),
            MLP(
                in_dim=x_dim,
                out_dim=hidden_dim,
                hidden_dim=hidden_dim,
            ),
            LocScale(
                in_dim=hidden_dim,
                out_dim=z_dim,
            ),
        )

        self.decode = nn.Sequential(
            MLP(
                in_dim=z_dim,
                out_dim=hidden_dim,
                hidden_dim=hidden_dim,
            ),
            Binary(
                in_dim=hidden_dim,
                out_dim=x_dim,
            ),
        )

        if use_cuda:
            self.cuda()

    def model(self, x: Tensor) -> None:
        """
        Generative model p(x|z)p(z).
        Describes the generative story of our data.
        """
        pyro.module("decode", self.decode)
        N = x.shape[0]
        with pyro.plate("N", N):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            # sample latent variable z.
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # decode and sample observation
            # validate_args=False to allow for pi in [0, 1]
            pi = self.decode(z)
            pyro.sample(
                "x",
                dist.Bernoulli(pi, validate_args=False).to_event(1),
                obs=x.view(N, -1),
            )

    def guide(self, x: Tensor) -> None:
        """
        Variational distribution q(z|x).
        Used to infer the latent variables in our model.
        For a VAE this is just a neural network.
        """
        pyro.module("encode", self.encode)
        with pyro.plate("N", x.shape[0]):
            z_loc, z_scale = self.encode(x)
            p_z = dist.Normal(z_loc, z_scale).to_event(1)
            pyro.sample("z", p_z)

    def reconstruct_img(self, x: Tensor):
        # encode image x
        z_loc, z_scale = self.encode(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decode(z)
        return loc_img

        import time


import traceback


def make_vae(
    data_set,
    z_dim,
    hidden_dims,
    lr=1.0e-3,
    epochs=101,
    vae_class=AmortizedVAE,
    batch_size=128,
    cuda=True,
    should_save=True,
    aux_loss_multiplier=None,
):
    # vae = VAE(use_cuda=True, z_dim=64 * 32, hidden_dim=64 * 48)
    params = {}
    if aux_loss_multiplier is not None:
        params["aux_loss_multiplier"] = aux_loss_multiplier
    vae = vae_class(
        use_cuda=cuda, z_dim=z_dim, hidden_dim=hidden_dims, x_dim=X * Y, **params
    )

    ds = torch.utils.data.TensorDataset(data_set)
    kwargs = {
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
        "batch_size": batch_size,
    }
    train_ds, valid_ds = torch.utils.data.random_split(ds, (0.9, 0.1))
    train_dl = torch.utils.data.DataLoader(train_ds, **kwargs)
    valid_dl = torch.utils.data.DataLoader(valid_ds, **kwargs)

    optimizer = Adam({"lr": lr})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    def train(svi, train_loader, use_cuda=False):
        # initialize loss accumulator
        epoch_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for i, x in enumerate(train_loader):
            x = x[0]
            if cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        return total_epoch_loss_train

    def evaluate(svi, test_loader, use_cuda=False):
        # initialize loss accumulator
        test_loss = 0.0
        # compute the loss over the entire test set
        for x in test_loader:
            x = x[0]
            # if on GPU put mini-batch into CUDA memory
            if cuda:
                x = x.cuda()
            test_loss += svi.evaluate_loss(x)
        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test
        return total_epoch_loss_test

    TEST_FREQUENCY = 10
    train_elbo = []
    test_elbo = []

    #     try:
    # training loop
    for epoch in range(epochs):
        total_epoch_loss_train = train(svi, train_dl, use_cuda=cuda)
        train_elbo.append(-total_epoch_loss_train)
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, valid_dl, use_cuda=cuda)
            test_elbo.append(-total_epoch_loss_test)
            print(
                "[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
    #     except Exception as e:
    #         print(traceback.format_exc())

    if should_save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_file = (
            f"{timestr}_{vae_class.__name__}_{data_set.shape[0]}_{z_dim}_{hidden_dims}"
        )
        save({"vae": vae, "train_elbo": train_elbo, "test_elbo": test_elbo}, model_file)
        pyro.get_param_store().save(f"{model_file}_params")

    return vae, train_elbo, test_elbo


class CategoricalLayer(nn.Module):
    def __init__(self, in_dim: int, x_dim: int, categories_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, x_dim * categories_dim)
        # Massage into probabilities.
        self.softmax = nn.Softmax(dim=2)
        self.in_dim = in_dim
        self.x_dim = x_dim
        self.categories_dim = categories_dim

    def forward(self, z: Tensor) -> Tensor:
        hidden = self.linear(z)

        # batch_size x x_dim x categories_dim
        hidden_reshaped = hidden.resize(z.shape[0], self.x_dim, self.categories_dim)

        # Normalize each x to represent the params for a Categorical
        pi_s = self.softmax(hidden_reshaped)
        return pi_s


class AmortizedVAECategorical(nn.Module):
    """
    Standard variational encoder model
    for categorical image data with a
    Gaussian latent variable.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int,
        use_cuda: bool,
        categories_dim: int = 5,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.categories_dim = 5

        embedding_dim = 1
        self.encode = nn.Sequential(
            nn.Flatten(),
            nn.Embedding(categories_dim, embedding_dim),
            nn.Flatten(),
            MLP(
                in_dim=x_dim * embedding_dim,
                out_dim=hidden_dim,
                hidden_dim=hidden_dim,
            ),
            LocScale(
                in_dim=hidden_dim,
                out_dim=z_dim,
            ),
        )

        self.decode = nn.Sequential(
            MLP(
                in_dim=z_dim,
                out_dim=hidden_dim,
                hidden_dim=hidden_dim,
            ),
            CategoricalLayer(
                in_dim=hidden_dim, x_dim=x_dim, categories_dim=categories_dim
            ),
        )

        if use_cuda:
            self.cuda()

    def model(self, x: Tensor) -> None:
        """
        Generative model p(x|z)p(z).
        Describes the generative story of our data.
        """
        pyro.module("decode", self.decode)
        N = x.shape[0]
        with pyro.plate("N", N):
            z_loc = x.new_zeros(torch.Size((N, self.z_dim)))
            z_scale = x.new_ones(torch.Size((N, self.z_dim)))

            # sample latent variable z.
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # decode and sample observation
            # validate_args=False to allow for pi in [0, 1]
            pis = self.decode(z)

            pyro.sample(
                "x", dist.Categorical(pis).to_event(1), obs=x.view(N, self.x_dim)
            )

    def guide(self, x: Tensor) -> None:
        """
        Variational distribution q(z|x).
        Used to infer the latent variables in our model.
        For a VAE this is just a neural network.
        """
        pyro.module("encode", self.encode)
        with pyro.plate("N", x.shape[0]):
            z_loc, z_scale = self.encode(x)
            p_z = dist.Normal(z_loc, z_scale).to_event(1)
            pyro.sample("z", p_z)

    def reconstruct_img(self, x: Tensor):
        # encode image x
        z_loc, z_scale = self.encode(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decode(z)
        return np.argmax(loc_img, axis=1)


def load_model(file: str):
    f = load(file)
    vae = f["vae"]
    train_elbos = f["train_elbo"]
    test_elbo = f["test_elbo"]
    pyro.get_param_store().load(f"{file}_params")
    return vae, train_elbos, test_elbo
