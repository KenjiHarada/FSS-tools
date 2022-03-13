"""Utility codes for the FSS analysis by GP in torch.

This file includes two classes and a function to do the FSS analysis by GP in torch.
"""
import torch
import gpytorch
import torch.nn as nn
from . import fss


class ExactGPModel(gpytorch.models.ExactGP):
    """Simple exact GP model.
    
    This is the most simplest GP model.

    Args:
        train_x (torch.Tensor): x-coordinate of a training data
        train_y (torch.Tensor): y-coordinate of a training data
        likelihood (gpytorch.likelihoods.Likelihood): likelihood of GP
    """

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, X):
        """Calculate a distribution as GP."""
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP(nn.Module):
    """GP model for FSS.
    
    This is a specialized GP model for FSS. The scaling function is defined as a sample generated from a GP.

    Reference:
        Kenji Harada: Bayesian inference in the scaling analysis of critical phenomena,
        Physical Review E 84 (2011) 056704.
        DOI: 10.1103/PhysRevE.84.056704 (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.056704)

    Args:
        with_statistical_error (bool): If True(default), use a statistical error in GP for FSS.

    Attributes:
        with_statistical_error (bool): If True(default), use a statistical error in GP for FSS.
    """

    def __init__(self, with_statistical_error=True):
        super(GP, self).__init__()
        dummy = torch.zeros(1, dtype=torch.float32)
        self.with_statistical_error = with_statistical_error
        if self.with_statistical_error:
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                dummy, learn_additional_noise=True
            )
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(dummy, dummy, self.likelihood)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def train(self):
        """Move to a train mode."""
        self.model.train()
        self.likelihood.train()

    def eval(self):
        """Move to an eval mode."""
        self.model.eval()
        self.likelihood.eval()

    def loss(self, X, Y, E=None):
        """Calculate a loss.
        
        We set a new train data (X, Y, E) in GP and calculate a loss function.

        Args:
            X, Y, E (torch.Tensor): train data for GP. If E is None, we don't use a statistical error.

        Return:
            loss (torch.Tensor):
        """
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        self.model.set_train_data(inputs=X, targets=Y, strict=False)
        if self.with_statistical_error:
            E = E.reshape(-1)
            self.likelihood.noise = E * E
        output = self.model(X)
        loss = -self.mll(output, Y)
        return loss

    def prediction(self, X):
        """Calculate a prediction.
        
        We calculate a predicted distribution of values for given points.

        Args:
            X (torch.Tensor): x-coordinate of points

        Return:
            observed_pred (gpytorch.Distribution): a predicted distribution (multivariate normal) of values
        """
        X = X.reshape(-1)
        if self.with_statistical_error:
            sigma2 = torch.zeros(len(X))
            observed_pred = self.likelihood(self.model(X), noise=sigma2)
        else:
            observed_pred = self.likelihood(self.model(X))
        return observed_pred


def do_fss(
    dataset: fss.Dataset,
    model: GP,
    optimizer: torch.optim.Optimizer,
    transform: fss.Transform = None,
    num_epochs=5000,
):
    """Do a FSS analysis by GP.

    This is an example code to do a FSS based on GP.
    The scaling form is defined as A = L^(-c2) f[(T-tc) * L^c1].
    Parameter tc is critical temperature, c1 and c2 are critical exponents.

    Reference:
        Kenji Harada: Bayesian inference in the scaling analysis of critical phenomena,
        Physical Review E 84 (2011) 056704.
        DOI: 10.1103/PhysRevE.84.056704 (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.056704)

    Args:
        dataset (fss.Dataset): dataset for FSS
        model (GP): GP model of a scaling function
        optimizer (torch.optim): optimizer
        transform (fss.Transform or None): FSS transformation. If None, no transformation.
        num_epochs (int): the number of epochs
    
    Return:
        [tc, c1] or [tc, c1, c2]:
    """
    model.train()
    if transform is None:
        X = dataset.data[:, 1]
        Y = dataset.data[:, 2]
        if model.with_statistical_error:
            E = dataset.data[:, 3]
        else:
            E = None
        for _ in range(num_epochs):
            optimizer.zero_grad()
            loss = model.loss(X, Y, E)
            loss.backward()
            optimizer.step()
        return
    else:
        for _ in range(num_epochs):
            optimizer.zero_grad()
            new_data = transform(dataset.data)
            X = new_data[:, 0]
            Y = new_data[:, 1]
            if model.with_statistical_error:
                E = new_data[:, 2]
            else:
                E = None
            loss = model.loss(X, Y, E)
            loss.backward()
            optimizer.step()
        if transform.no_c2:
            return [transform.tc, transform.c1]
        else:
            return [transform.tc, transform.c1, transform.c2]
