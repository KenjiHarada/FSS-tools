"""Utility codes for the FSS analysis by DNN in torch.

This file includes two classes and a function to do the FSS analysis by DNN in torch.
"""
import torch
import torch.nn as nn
from . import fss


class Rational(nn.Module):
    """Rational activation function.

    Activation function, f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function

    Reference:
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """

    def __init__(self):
        super().__init__()
        self.coeffs = nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the coefficients of P and Q."""
        self.coeffs.data = torch.Tensor(
            [[1.1915, 0.0], [1.5957, 2.383], [0.5, 0.0], [0.0218, 1.0]]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate an output of Rational neural networks.
        
        Args:
            input (torch.Tensor):
        
        Return:
            output (torch.Tensor):
        """
        self.coeffs.data[0, 1].zero_()
        exp = torch.tensor([3.0, 2.0, 1.0, 0.0], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


class MLP(nn.Module):
    """Multi layer perceptron for FSS.

    Note:
        Input and output dimension are one.
    """

    def __init__(
        self, hidden_sizes=[20, 20, 20], act=Rational,
    ):
        """Initialize a MLP.

        Args:
            hidden_sizes ([int]): the shape of layers
            act (nn.Module): activation function
        """
        super(MLP, self).__init__()
        layer = [
            nn.Linear(1, hidden_sizes[0]),
            act(),
        ]
        for i in range(len(hidden_sizes) - 1):
            layer.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layer.append(act())
        layer.append(nn.Linear(hidden_sizes[-1], 1))
        self.layer = nn.ModuleList(layer)

    def forward(self, x):
        """Calculate an output of MLP."""
        for i in range(len(self.layer)):
            x = self.layer[i](x)
        return x


def get_column(data: torch.Tensor, i):
    """get a column from a tensor.
    
    Args:
        data (torch.Tensor): a matrix
        i (int): the index of column
    Return:
        column (torch.Tensor): a vector with a shape (:, 1).
    """
    d = data.size(dim=-1)
    x = data.view(-1, d)
    return x[:, i].view(-1, 1)


def do_fss(
    dataset: fss.Dataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    transform: fss.Transform = None,
    num_epochs=10000,
    use_MSE=False,
):
    """Do a FSS analysis.

    The scaling form is defined as A = L^(-c2) f[(T-tc) * L^c1].
    Parameter tc is critical temperature, c1 and c2 are critical exponents.

    Args:
        dataset (fss.Dataset): dataset
        model (nn.Module): model of the scaling function
        optimizer (torch.optim): optimizer
        transform (fss.Transform or None): FSS transformation. If None, no transformation.
        num_epochs (int): the number of epochs
    
    Return:
        [tc, c1] or [tc, c1, c2] : tc is critical temperature, c1 and c2 are critical exponents.
    """

    # optimization
    if not use_MSE and dataset.data.size(dim=-1) == 4:
        loss_fn = nn.GaussianNLLLoss()
        with_error = True
    else:
        loss_fn = nn.MSELoss()
        with_error = False
    for _ in range(num_epochs):
        optimizer.zero_grad()
        if transform is None:
            new_data = dataset.data[:, 1:]
        else:
            new_data = transform(dataset.data)
        X = get_column(new_data, 0)
        Y = get_column(new_data, 1)
        if with_error:
            E = get_column(new_data, 2)
            loss = loss_fn(model(X), Y, E * E)
        else:
            loss = loss_fn(model(X), Y)
        loss.backward()
        optimizer.step()
    if transform is None:
        return
    if transform.no_c2:
        return [transform.tc, transform.c1]
    else:
        return [transform.tc, transform.c1, transform.c2]
