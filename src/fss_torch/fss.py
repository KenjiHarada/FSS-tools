"""
Classes for FSS analysis in torch.

This file defines two classes for the FSS analysis in torch: Dataset and Transform.
"""

import math
import numpy as np
import torch
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):
    """Dataset for finite-size scaling (FSS) analysis.

    Each data is (system size, temperature, observable, statistical error)
    or (system size, temperature, observable).
    
    Note: All data is rescaled as follows. The rescaled maximum system size is one,
        the range of rescaled temperatures for maximum system size = [-1, 1],
        and the width of the range of rescaled observables is one.

    Args:
        data (numpy): It includes (L, T, A, E) or (L, T, A).
            L is the first column, T is the second one,
            A is the third one, E is the four-th column if there exists.
        rescale (bool): If True(default), all data are rescaled.
   
    Attributes: 
        data (numpy): It includes (L, T, A, E) or (L, T, A).
            L is the first column, T is the second one,
            A is the third one, E is the four-th column if there exists.
        scale (float): the rescaled A = (A / scale)
    """

    def __init__(self, data: np.ndarray, rescale=True):
        super(Dataset, self).__init__()
        self.data = torch.Tensor(data)
        self.max_system_size = data[:, 0].max()
        if rescale:
            idx = data[:, 0] == self.max_system_size
            tmax, tmin = data[idx, 1].min(), data[idx, 1].max()
            self.t_middle = (tmax + tmin) / 2.0
            self.t_scale = (tmax - tmin) / 2.0
            self.scale = data[:, 2].max() - data[:, 2].min()
            self.data[:, 0] = self.data[:, 0] / self.max_system_size
            self.data[:, 1] = (self.data[:, 1] - self.t_middle) / self.t_scale
            self.data[:, 2:] = self.data[:, 2:] / self.scale
        else:
            self.t_middle = 0
            self.t_scale = 1.0
            self.scale = 1.0

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.data.size(dim=0)

    def transform_t(self, x):
        return (x - self.t_middle) / self.t_scale

    def inv_transform_t(self, x):
        return x * self.t_scale + self.t_middle

    @classmethod
    def fromFile(cls, fname, only_maxsize=False):
        """Load a data for FSS from a file.

        The line in a data file consists of (L, T, A, E) or (L, T, A).

        Args:
            fname (str): the path of a data file
            only_maxsize (bool): if True(not default), load only data for maximum system size.
    
        Returns:
            dataset (fss.Dataset):
        """
        data = np.loadtxt(fname=fname, dtype=np.float32)
        if only_maxsize:
            max_system_size = data[:, 0].max()
            idx = data[:, 0] == max_system_size
            data = data[idx, :]
        return cls(data)

    def random_choice(self, rate):
        n0 = self.data.size(dim=0)
        n = int(n0 * rate)
        rng = np.random.default_rng()
        x = rng.choice(self.data.numpy(), n, replace=False, shuffle=False)
        new_dataset = Dataset(x, rescale=False)
        new_dataset.t_middle = self.t_middle
        new_dataset.t_scale = self.t_scale
        new_dataset.scale = self.scale
        new_dataset.max_system_size = self.max_system_size
        return new_dataset


class Transform(nn.Module):
    """Finite-size scaling transformation.
    
    The scaling form is defined as A = L^(-c2) f[(T-tc) * L^c1].
    Parameter tc is critical temperature, c1 and c2 are critical exponents.

    Args:
        dataset (Dataset): t_middle and t_scale are necessary.
        initial_valuse ([tc, c1] or [tc, c1, c2]): initial values of parameter
        mask (None, [bool, bool] or [bool, bool, bool]): if True, the value is fixed.

    Attributes:
        tc : critical temperature
        c1, c2 : critical exponent
        sigma2 : noise
    """

    def _softplus_inv(self, x):
        if x > 20:
            return x
        else:
            return math.log(math.exp(x) - 1)

    def __init__(
        self, initial_values, mask=None, add_noise=False,
    ):
        super(Transform, self).__init__()
        raw_tc = initial_values[0]
        raw_c1 = self._softplus_inv(initial_values[1])
        if len(initial_values) == 2:
            self.no_c2 = True
            self.initial_values = torch.Tensor([raw_tc, raw_c1])
        else:
            self.no_c2 = False
            self.initial_values = torch.Tensor([raw_tc, raw_c1, initial_values[2]])
        self.params = nn.Parameter(torch.Tensor(self.initial_values))
        self.mask = mask
        if add_noise:
            self.raw_sigma2 = nn.Parameter(torch.Tensor([0.0,]))
        else:
            self.raw_sigma2 = None

    @property
    def tc(self):
        return self.params[0].item()

    @property
    def c1(self):
        return torch.nn.functional.softplus(self.params[1]).item()

    @property
    def c2(self):
        return self.params[2].item()

    @property
    def sigma2(self):
        if self.raw_sigma2 is None:
            return 0e0
        else:
            return torch.nn.functional.softplus(self.raw_sigma2).item()

    # scaling transformation
    def scale_X(self, Ts, Ls, tc, c1):
        """Calculate a x-coordinate of a scaling function as (Ts - tc) * Ls^c1 + tc."""
        return (Ts - tc) * Ls.pow(c1) + tc

    def scale_A(self, As, Ls, c2):
        """Calculate a y-coordinate of a scaling function as As * Ls^c2."""
        return As * Ls.pow(c2)

    def forward(self, data):
        """Calculate transformed values.

        Args:
            data (torch.Tensor): Each row consists of (L, T, A) or (L, T, A, E)

        Returns:
            new_data (torch.Tensor): Each row consists of (X, Y) or (X, Y, E)
        """
        new_data = torch.zeros((data.size(dim=0), data.size(dim=-1) - 1))
        if self.no_c2:
            tc, raw_c1 = self.params
            if not self.mask is None:
                if self.mask[0]:
                    tc = self.initial_values[0]
                if self.mask[1]:
                    raw_c1 = self.initial_values[1]
            c1 = torch.nn.functional.softplus(raw_c1)
            new_data[:, 1] = data[:, 2]
            if data.size(dim=-1) == 4:
                if self.raw_sigma2 is None:
                    new_data[:, 2] = data[:, 3]
                else:
                    sigma2 = torch.nn.functional.softplus(self.raw_sigma2)
                    new_data[:, 2] = torch.sqrt(data[:, 3] * data[:, 3] + sigma2)
        else:
            tc, raw_c1, c2 = self.params
            if not self.mask is None:
                if self.mask[0]:
                    tc = self.initial_values[0]
                if self.mask[1]:
                    raw_c1 = self.initial_values[1]
                if self.mask[2]:
                    c2 = self.initial_values[2]
            c1 = torch.nn.functional.softplus(raw_c1)
            new_data[:, 1] = self.scale_A(data[:, 2], data[:, 0], c2)
            if data.size(dim=-1) == 4:
                if self.raw_sigma2 is None:
                    new_data[:, 2] = self.scale_A(data[:, 3], data[:, 0], c2)
                else:
                    sigma2 = torch.nn.functional.softplus(self.raw_sigma2)
                    new_data[:, 2] = self.scale_A(
                        torch.sqrt(data[:, 3] * data[:, 3] + sigma2), data[:, 0], c2
                    )
        new_data[:, 0] = self.scale_X(data[:, 1], data[:, 0], tc, c1)
        return new_data
