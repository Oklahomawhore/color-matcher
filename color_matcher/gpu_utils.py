#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
    Copyright (c) 2020 Christopher Hahne <info@christopherhahne.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> 'torch.device':
    """
    Detect the best available compute device with three-level fallback:
    CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.

    :return: torch.device object for the best available backend
    :rtype: torch.device
    """
    if not TORCH_AVAILABLE:
        raise ImportError('PyTorch is required for GPU acceleration. Install with: pip install torch>=2.1')

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_type() -> str:
    """
    Return the device type string: 'cuda', 'mps', or 'cpu'.

    :return: Device type string
    :rtype: str
    """
    if not TORCH_AVAILABLE:
        return 'cpu'
    return get_device().type


# Module-level constants for quick checks
DEVICE_TYPE = get_device_type() if TORCH_AVAILABLE else 'cpu'
HAS_GPU = DEVICE_TYPE != 'cpu'


def numpy_to_tensor(arr: np.ndarray, device: 'torch.device' = None, dtype=None) -> 'torch.Tensor':
    """
    Convert a NumPy array to a PyTorch tensor on the specified device.

    On MPS (Apple Silicon unified memory), the transfer is near zero-copy.
    On CUDA, data is copied to GPU VRAM.

    :param arr: Input NumPy array
    :param device: Target device (defaults to best available)
    :param dtype: Optional torch dtype override (defaults to float32)

    :type arr: :class:`~numpy:numpy.ndarray`
    :type device: :class:`torch.device`
    :type dtype: :class:`torch.dtype`

    :return: PyTorch tensor on the specified device
    :rtype: torch.Tensor
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = torch.float32

    # Ensure contiguous array for efficient conversion
    arr = np.ascontiguousarray(arr)

    # Convert to float for processing if integer type
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)

    return torch.from_numpy(arr).to(dtype=dtype, device=device)


def tensor_to_numpy(t: 'torch.Tensor') -> np.ndarray:
    """
    Convert a PyTorch tensor back to a NumPy array.

    :param t: Input tensor (on any device)
    :type t: :class:`torch.Tensor`

    :return: NumPy array (on CPU)
    :rtype: np.ndarray
    """
    return t.detach().cpu().numpy()


def ensure_float32(t: 'torch.Tensor') -> 'torch.Tensor':
    """
    Ensure tensor is float32 (MPS does not fully support float64).

    :param t: Input tensor
    :type t: :class:`torch.Tensor`
    :return: float32 tensor
    :rtype: torch.Tensor
    """
    if t.dtype != torch.float32:
        return t.to(torch.float32)
    return t


@torch.no_grad()
def gpu_cov(x: 'torch.Tensor') -> 'torch.Tensor':
    """
    Compute covariance matrix on GPU, equivalent to np.cov().
    Input x has shape (C, N) where C is channels and N is number of pixels.

    :param x: Input tensor of shape (C, N)
    :type x: :class:`torch.Tensor`

    :return: Covariance matrix of shape (C, C)
    :rtype: torch.Tensor
    """
    n = x.shape[1]
    mean = x.mean(dim=1, keepdim=True)
    x_centered = x - mean
    cov = x_centered @ x_centered.T / (n - 1)
    return cov


@torch.no_grad()
def gpu_interp(x: 'torch.Tensor', xp: 'torch.Tensor', fp: 'torch.Tensor') -> 'torch.Tensor':
    """
    GPU-based linear interpolation, equivalent to np.interp().
    Uses torch.searchsorted for bin finding and linear interpolation.

    :param x: x-coordinates at which to evaluate the interpolation
    :param xp: x-coordinates of data points (must be increasing)
    :param fp: y-coordinates of data points

    :type x: :class:`torch.Tensor`
    :type xp: :class:`torch.Tensor`
    :type fp: :class:`torch.Tensor`

    :return: Interpolated values
    :rtype: torch.Tensor
    """
    # Clamp to range of xp
    x_clamped = torch.clamp(x, xp[0], xp[-1])

    # Find insertion indices
    indices = torch.searchsorted(xp, x_clamped).clamp(1, len(xp) - 1)

    # Linear interpolation
    x_low = xp[indices - 1]
    x_high = xp[indices]
    f_low = fp[indices - 1]
    f_high = fp[indices]

    # Weight computation with safe division
    denom = x_high - x_low
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    weight = (x_clamped - x_low) / denom

    return f_low + weight * (f_high - f_low)
