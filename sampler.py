""" Sampler of points from UV space.

Author: Jan Bednarik, jan.bednarik@epfl.ch
Date: 17.2.2020
"""

# 3rd party
import torch
import torch.nn as nn

# Python std
from abc import ABC, abstractmethod

# Project files
from helpers import Device


class FNSampler(ABC, nn.Module, Device):
    """ Abstract base sampler class. """
    def __init__(self, gpu=True):
        ABC.__init__(self)
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError


class FNSampler2D(FNSampler):
    """ Abstract base class for sampling the 2D parametric space.

    Args:
        gpu (bool): Whether to use GPU.
        u_range (tuple): Range of u-axis, (u_min, u_max).
        v_range (tuple): Range of v-axis, (v_min, v_max).
    """
    def __init__(self, u_range, v_range, gpu=True):
        super(FNSampler2D, self).__init__(gpu=gpu)
        self.check_range(u_range)
        self.check_range(v_range)
        self._u_range = u_range
        self._v_range = v_range

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def check_range(r):
        """ Checks that the given range `r` (min, max) is a 2-tuple and
        max >= min.

        Args:
            r (tuple): 2-tuple, range, (min, max), max >= min.
        """
        assert(len(r) == 2)
        assert(r[1] >= r[0])


class FNSamplerRandUniform(FNSampler2D):
    """ Random 2D grid points generator.

    Args:
        u_range (tuple): Range of u-axis, (u_min, u_max).
        v_range (tuple): Range of v-axis, (v_min, v_max).
        num_samples (int): # samples.
        gpu (bool): Whether to use gpu.
    """
    def __init__(self, u_range, v_range, num_samples, gpu=True):
        super(FNSamplerRandUniform, self).__init__(u_range, v_range, gpu=gpu)
        self._num_samples = num_samples

    def forward(self, B, num_samples=None, u_range=None, v_range=None):
        """
        Args:
            B (int): Current batch size.

        Returns:
            torch.Tensor: Randomly sampled 2D points,
                shape (B, `num_samples`, 2).
        """
        ns = (num_samples, self._num_samples)[num_samples is None]
        ur = (u_range, self._u_range)[u_range is None]
        vr = (v_range, self._v_range)[v_range is None]

        return torch.cat(
            [torch.empty((B, ns, 1)).uniform_(*ur),
             torch.empty((B, ns, 1)).uniform_(*vr)], dim=2).to(self.device)
    

class FNSamplerRegularGrid(FNSampler2D):
    """ Regular 2D grid points generator. --zhantao

    Args:
        u_range (tuple): Range of u-axis, (u_min, u_max).
        v_range (tuple): Range of v-axis, (v_min, v_max).
        num_samples (int): # samples.
        gpu (bool): Whether to use gpu.
    """
    def __init__(self, u_range, v_range, num_samples, num_patches, gpu=True):
        super(FNSamplerRegularGrid, self).__init__(u_range, v_range, gpu=gpu)
        self._num_samples = num_samples
        self._num_patches = num_patches
        
    def forward(self, B, num_samples=None, 
                n_x = None, n_y = None,
                u_range=None, v_range=None):
        """
        Args:
            B (int): Current batch size.

        Returns:
            torch.Tensor: Randomly sampled 2D points,
                shape (B, `num_samples`, 2).
        """
        ns = torch.sqrt(torch.tensor((num_samples, self._num_samples)[num_samples is None]/self._num_patches))
        
        if n_x is not None and n_y is not None:
            ns_x = n_x
            ns_y = n_y
        elif ns.item().is_integer():
            ns_x = ns
            ns_y = ns
        else:
            ns_x = torch.tensor(10.0)
            ns_y = torch.tensor((num_samples, self._num_samples)[num_samples is None]/self._num_patches)//ns_x
            if not ns_y.item().is_integer():
                print("The number of samples (points), i.e. M, after being divided by the number of patches, should be n*10.")
                raise ValueError
        
        ur = (u_range, self._u_range)[u_range is None]
        vr = (v_range, self._v_range)[v_range is None]
        
        xs = torch.linspace(ur[0], ur[1], ns_x.int().item())
        ys = torch.linspace(vr[0], vr[1], ns_y.int().item())
        xg, yg = torch.meshgrid(xs, ys)        

        return torch.cat(
            [xg.flatten()[:,None],
             yg.flatten()[:,None]], dim=1)[None,:,:].repeat(B, self._num_patches , 1).to(self.device)
    
