from utils import MLP, Exp
import time
from save import save, load

import torch
import torch.nn as nn
from torch import Tensor

import pyro
import pyro.distributions as dist

from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import Adam, ClippedAdam


class CategoricalLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, categories_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * categories_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.categories_dim = categories_dim

        # Massage into probabilities.
        self.softmax = nn.Softmax(2)

    def forward(self, z: Tensor) -> Tensor:
        hidden = self.linear(z)

        # print(hidden.shape)

        # batch_size x x_dim x categories_dim
        hidden_reshaped = hidden.resize(z.shape[0], self.out_dim, self.categories_dim)

        # Normalize each x to represent the params for a Categorical
        pi_s = self.softmax(hidden_reshaped)
        return pi_s


class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset
    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """

    def __init__(
        self,
        input_size=784,
        z_dim=50,
        hidden_layers=(500,),
        config_enum=None,
        use_cuda=False,
        aux_loss_multiplier=None,
        categories_dim=None,
        bed_bath_dim=5,
    ):

        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == "parallel"
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        self.categories_dim = categories_dim
        self.embedding_dim = 1
        self.bed_bath_dim = bed_bath_dim

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter
        self.encoder_beds = nn.Sequential(
            MLP(
                [self.input_size] + hidden_sizes,
                activation=nn.Softplus,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            ),
            CategoricalLayer(
                in_dim=hidden_sizes[-1],
                out_dim=1,
                categories_dim=self.bed_bath_dim,
            ),
        )
        self.encoder_baths = nn.Sequential(
            MLP(
                [self.input_size] + hidden_sizes,
                activation=nn.Softplus,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            ),
            CategoricalLayer(
                in_dim=hidden_sizes[-1],
                out_dim=1,
                categories_dim=self.bed_bath_dim,
            ),
        )

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        self.encoder_z = MLP(
            [self.input_size + 2] + hidden_sizes + [[z_dim, z_dim]],
            activation=nn.Softplus,
            output_activation=[None, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # if self.categories_dim is not None:
        #     self.encoder_beds = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Embedding(self.categories_dim, self.embedding_dim),
        #         nn.Flatten(),
        #         self.encoder_beds,
        #     )
        #     self.encoder_baths = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Embedding(self.categories_dim, self.embedding_dim),
        #         nn.Flatten(),
        #         self.encoder_baths,
        #     )

        if self.categories_dim is not None:
            self.decoder = nn.Sequential(
                MLP(
                    [z_dim + 2] + hidden_sizes,
                    activation=nn.Softplus,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                ),
                CategoricalLayer(
                    in_dim=hidden_sizes[-1],
                    out_dim=self.input_size,
                    categories_dim=self.categories_dim,
                ),
            )
        else:
            self.decoder = MLP(
                # 2 Extra because of the beds, baths.
                [z_dim + 2] + hidden_sizes + [self.input_size],
                activation=nn.Softplus,
                output_activation=nn.Sigmoid,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, beds=None, baths=None):
        """
        The model corresponds to the following generative process:

        Latent:
        p(z) = normal(0,I)                                                        # Architecture style

        Semi Supervised:
        p(beds|x) = Normal(3, 2)                                                  # Number of beds
        p(baths|x) = Normal(3, 2)                                                 # Number of baths

        p(x|beds, baths, balconcy, z) = categorical(loc(beds, baths, balcony, z)) # A plan

        loc is given by a neural network  `decoder`
        :param xs: a batch of scaled vectors of pixels from an image

        :param beds: (optional) a batch of the # of beds.
        :param beds: (optional) a batch of the # of beds.
        :param balconies: (optional) a batch of if there are balconies

        :return: None
        """
        # print(beds.shape)
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("decoder", self.decoder)

        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate("N"):
            # sample the handwriting style from the constant prior distribution
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            # if the labels are supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            bed_baths_prior = torch.ones([batch_size, self.bed_bath_dim], **options) / (
                1.0 * self.bed_bath_dim
            )
            # print(bed_baths_prior.shape)
            # print(bed_baths_scale.shape)

            beds = pyro.sample(
                "beds",
                dist.Categorical(bed_baths_prior).to_event(1),
                obs=beds,
            )
            baths = pyro.sample(
                "baths",
                dist.Categorical(bed_baths_prior).to_event(1),
                obs=baths,
            )

            # Finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network. We disable validation
            # since the decoder output is a relaxed Bernoulli value.
            if self.categories_dim is None:
                loc = self.decoder.forward([zs, beds, baths])
                pyro.sample(
                    "x", dist.Bernoulli(loc, validate_args=False).to_event(1), obs=xs
                )
                # return the loc so we can visualize it later
                return loc
            else:
                pis = self.decoder.forward([zs, beds, baths])
                pyro.sample("x", dist.Categorical(pis).to_event(1), obs=xs)
                # return pis

    def guide(self, xs, beds=None, baths=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        pyro.module("encoder_beds", self.encoder_beds)
        pyro.module("encoder_baths", self.encoder_baths)
        pyro.module("encoder_z", self.encoder_z)
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            if beds is None:
                pis = self.encoder_beds.forward(xs)
                beds = pyro.sample("beds", dist.Categorical(pis))

            if baths is None:
                pis = self.encoder_baths.forward(xs)
                baths = pyro.sample("baths", dist.Categorical(pis))

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,beds,baths,balconies) = normal(loc(x,beds,baths,balconies),scale(x,beds,baths,balconies))
            loc, scale = self.encoder_z.forward([xs, beds, baths])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def model_classify(self, xs, beds=None, baths=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("encoder_beds", self.encoder_beds)
        pyro.module("encoder_baths", self.encoder_baths)
        # inform Pyro that the variables in the batch of xs, (beds, baths) are conditionally independent
        with pyro.plate("data"):
            # this here is the extra terms to yield an auxiliary loss that we do gradient descent on
            if baths is not None:
                pis = self.encoder_baths.forward(xs)
                pis = pis.resize(xs.shape[0], self.bed_bath_dim)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample(
                        "baths_aux",
                        dist.Categorical(pis),
                        obs=baths,
                    )

            if beds is not None:
                pis = self.encoder_beds.forward(xs)
                pis = pis.resize(xs.shape[0], self.bed_bath_dim)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample(
                        "beds_aux",
                        dist.Categorical(pis),
                        obs=beds,
                    )

    def guide_classify(self, xs, beds=None, baths=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


def setup_data_loaders(data_sets, batch_size):
    ds = torch.utils.data.TensorDataset(*data_sets)
    kwargs = {
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
        "batch_size": batch_size,
    }

    train_ds, valid_ds = torch.utils.data.random_split(ds, (0.9, 0.1))
    train_dl = torch.utils.data.DataLoader(train_ds, **kwargs)
    valid_dl = torch.utils.data.DataLoader(valid_ds, **kwargs)
    return train_dl, valid_dl


def make_vae(
    data_sets,
    z_dim,
    hidden_layers,
    lr=1.0e-3,
    epochs=101,
    vae_class=SSVAE,
    batch_size=128,
    cuda=True,
    should_save=True,
    aux_loss_multiplier=None,
    seed=None,
    categories_dim=None,
    bed_bath_dim=9,
):
    if seed is not None:
        pyro.set_rng_seed(seed)

    params = {}
    if aux_loss_multiplier is not None:
        params["aux_loss_multiplier"] = aux_loss_multiplier

    vae = vae_class(
        use_cuda=cuda,
        z_dim=z_dim,
        hidden_layers=hidden_layers,
        input_size=64 * 64,
        categories_dim=categories_dim,
        bed_bath_dim=bed_bath_dim,
        **params,
    )

    optimizer = Adam({"lr": lr})
    optimizer_clipped = ClippedAdam({"lr": lr})
    losses = [SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())]

    if aux_loss_multiplier is not None:
        losses.append(
            SVI(
                vae.model_classify,
                vae.guide_classify,
                optimizer_clipped,
                loss=Trace_ELBO(),
            )
        )

    train_dl, valid_dl = setup_data_loaders(data_sets, batch_size)

    def train(losses, train_loader, use_cuda=False):
        # initialize loss accumulator
        epoch_losses = [0.0] * len(losses)
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for xs, beds, baths in train_loader:
            for i, loss in enumerate(losses):
                if use_cuda:
                    xs = xs.cuda()
                    beds = beds.cuda()
                    baths = baths.cuda()
                # do ELBO gradient and accumulate loss
                epoch_losses[i] += loss.step(xs, beds, baths)

        return [loss / len(train_loader.dataset) for loss in epoch_losses]

    def evaluate(losses, test_loader, use_cuda=False):
        # initialize loss accumulator
        test_losses = [0.0] * len(losses)
        # compute the loss over the entire test set
        for xs, beds, baths in test_loader:
            for i, loss in enumerate(losses):
                if use_cuda:
                    xs = xs.cuda()
                    beds = beds.cuda()
                    baths = baths.cuda()
                test_losses[i] += loss.evaluate_loss(xs, beds, baths)

        return [loss / len(test_loader.dataset) for loss in test_losses]

    TEST_FREQUENCY = 10
    train_elbos = [[] for _ in losses]
    test_elbo = [[] for _ in losses]

    #     try:
    # training loop
    for epoch in range(epochs):
        total_epoch_loss_train = train(losses, train_dl, use_cuda=cuda)
        for i, loss in enumerate(total_epoch_loss_train):
            train_elbos[i].append(-loss)
            print("[epoch %03d]  average training loss: %.4f" % (epoch, loss))

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(losses, valid_dl, use_cuda=cuda)
            for i, loss in enumerate(total_epoch_loss_test):
                test_elbo[i].append(-loss)
                print("[epoch %03d] average test loss: %.4f" % (epoch, loss))
    #     except Exception as e:
    #         print(traceback.format_exc())

    if should_save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_file = f"{timestr}_{vae_class.__name__}_{data_sets[0].shape[0]}_{z_dim}_{'_'.join(map(str, hidden_layers))}"
        save(
            {"vae": vae, "train_elbo": train_elbos, "test_elbo": test_elbo}, model_file
        )
        pyro.get_param_store().save(f"{model_file}_params")

    return vae, train_elbos, test_elbo


def load_model(file: str):
    f = load(file)
    vae = f["vae"]
    train_elbos = f["train_elbo"]
    test_elbo = f["test_elbo"]
    pyro.get_param_store().load(f"{file}_params")
    return vae, train_elbos, test_elbo
