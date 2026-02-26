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
import torch
from types import FunctionType

from .mvgd_matcher import TransferMVGD
from .gpu_utils import get_device, numpy_to_tensor, tensor_to_numpy, ensure_float32, gpu_cov


class GPUTransferMVGD(TransferMVGD):
    """
    GPU-accelerated Multi-Variate Gaussian Distribution color transfer using PyTorch.
    Supports CUDA, MPS (Apple Silicon), and CPU fallback.
    """

    def __init__(self, *args, **kwargs):
        super(GPUTransferMVGD, self).__init__(*args, **kwargs)
        self._device = get_device()

        # GPU tensor versions of variables
        self._t_r = None
        self._t_z = None
        self._t_cov_r = None
        self._t_cov_z = None
        self._t_mu_r = None
        self._t_mu_z = None
        self._t_transfer_mat = None

        # Cached reference stats for video processing
        self._ref_stats_cached = False
        self._cached_t_cov_z = None
        self._cached_t_mu_z = None

    @torch.no_grad()
    def init_vars_gpu(self):
        """
        Initialize variables on GPU: reshape images, compute covariance and mean.
        """

        # Reshape source and reference images to (C, N) pixel matrices
        src_flat = self._src.reshape([-1, self._src.shape[2]]).T   # (C, N)
        ref_flat = self._ref.reshape([-1, self._ref.shape[2]]).T   # (C, N)

        # Transfer to GPU
        self._t_r = numpy_to_tensor(src_flat, self._device)
        self._t_z = numpy_to_tensor(ref_flat, self._device)

        # Compute covariance matrices on GPU
        self._t_cov_r = gpu_cov(self._t_r)
        self._t_cov_z = gpu_cov(self._t_z)

        # Compute means on GPU
        self._t_mu_r = self._t_r.mean(dim=1, keepdim=True)
        self._t_mu_z = self._t_z.mean(dim=1, keepdim=True)

        # Also set numpy versions for compatibility with check_dims
        self.r = tensor_to_numpy(self._t_r)
        self.z = tensor_to_numpy(self._t_z)
        self.cov_r = tensor_to_numpy(self._t_cov_r)
        self.cov_z = tensor_to_numpy(self._t_cov_z)
        self.mu_r = tensor_to_numpy(self._t_mu_r)
        self.mu_z = tensor_to_numpy(self._t_mu_z)

        # Validate dimensionality
        self.check_dims()

    @torch.no_grad()
    def init_vars_gpu_cached(self):
        """
        Initialize source variables on GPU, reusing cached reference stats (for video).
        """

        src_flat = self._src.reshape([-1, self._src.shape[2]]).T
        self._t_r = numpy_to_tensor(src_flat, self._device)
        self._t_cov_r = gpu_cov(self._t_r)
        self._t_mu_r = self._t_r.mean(dim=1, keepdim=True)

        # Use cached reference stats
        self._t_cov_z = self._cached_t_cov_z
        self._t_mu_z = self._cached_t_mu_z

        # Set numpy versions for check_dims compatibility
        self.r = tensor_to_numpy(self._t_r)
        self.cov_r = tensor_to_numpy(self._t_cov_r)
        self.cov_z = tensor_to_numpy(self._t_cov_z)
        self.mu_r = tensor_to_numpy(self._t_mu_r)
        self.mu_z = tensor_to_numpy(self._t_mu_z)

        self.check_dims()

    @torch.no_grad()
    def cache_ref_stats(self, ref: np.ndarray):
        """
        Pre-compute and cache reference image statistics for video frame reuse.
        Call this once before processing multiple frames.

        :param ref: Reference image
        :type ref: :class:`~numpy:numpy.ndarray`
        """
        ref_flat = ref.reshape([-1, ref.shape[2]]).T
        t_z = numpy_to_tensor(ref_flat, self._device)
        self._cached_t_cov_z = gpu_cov(t_z)
        self._cached_t_mu_z = t_z.mean(dim=1, keepdim=True)
        self._ref_stats_cached = True

    @torch.no_grad()
    def multivar_transfer_gpu(self, src: np.ndarray = None, ref: np.ndarray = None,
                               fun: FunctionType = None) -> np.ndarray:
        """
        GPU-accelerated MVGD color transfer.

        :param src: Source image that requires transfer
        :param ref: Palette image which serves as reference
        :param fun: Optional transfer function solver
        :type src: :class:`~numpy:numpy.ndarray`
        :type ref: :class:`~numpy:numpy.ndarray`
        :return: Resulting image after the MVGD mapping
        :rtype: np.ndarray
        """

        # Override source and reference image with arguments (if provided)
        self._src = src if src is not None else self._src
        self._ref = ref if ref is not None else self._ref

        # Check color channels
        self.validate_color_chs()

        # Initialize variables on GPU (use cached ref stats if available)
        if self._ref_stats_cached:
            self.init_vars_gpu_cached()
        else:
            self.init_vars_gpu()

        # Set solver function
        self._fun_call = fun if fun is FunctionType else self._fun_call

        # Compute transfer matrix on GPU
        self._t_transfer_mat = self._mkl_solver_gpu()

        # Apply transfer: res = T @ (r - mu_r) + mu_z
        res_t = torch.mm(self._t_transfer_mat, self._t_r - self._t_mu_r) + self._t_mu_z

        # Convert back to numpy and reshape
        res = tensor_to_numpy(res_t).T.reshape(self._src.shape)

        return res

    @torch.no_grad()
    def _mkl_solver_gpu(self) -> 'torch.Tensor':
        """
        GPU-accelerated MKL solver for the transfer matrix.
        Falls back to CPU for eigen decomposition if MPS doesn't support it.

        :return: Transfer matrix as GPU tensor
        :rtype: torch.Tensor
        """

        self.check_dims()

        cov_r = self._t_cov_r
        cov_z = self._t_cov_z

        # Eigen decomposition - may need CPU fallback on MPS
        try:
            eig_vals_r, eig_vecs_r = torch.linalg.eig(cov_r)
            eig_vals_r = eig_vals_r.real
            eig_vecs_r = eig_vecs_r.real
        except RuntimeError:
            # Fallback to CPU for eigendecomposition (3x3 matrix - trivial cost)
            cov_r_cpu = cov_r.cpu()
            eig_vals_r, eig_vecs_r = torch.linalg.eig(cov_r_cpu)
            eig_vals_r = eig_vals_r.real.to(self._device)
            eig_vecs_r = eig_vecs_r.real.to(self._device)

        # Clamp negative eigenvalues
        eig_vals_r = torch.clamp(eig_vals_r, min=0)

        # Reverse order (ascending -> descending)
        idx_r = torch.arange(eig_vals_r.shape[0] - 1, -1, -1, device=self._device)
        val_r = torch.diag(torch.sqrt(eig_vals_r[idx_r]))
        vec_r = eig_vecs_r[:, idx_r]

        # Inverse of val_r with numerical stability
        inv_r = torch.diag(1.0 / (torch.diag(val_r) + torch.finfo(torch.float32).eps))

        # Compute intermediate matrix
        mat_c = val_r @ vec_r.T @ cov_z @ vec_r @ val_r

        try:
            eig_vals_c, eig_vecs_c = torch.linalg.eig(mat_c)
            eig_vals_c = eig_vals_c.real
            eig_vecs_c = eig_vecs_c.real
        except RuntimeError:
            mat_c_cpu = mat_c.cpu()
            eig_vals_c, eig_vecs_c = torch.linalg.eig(mat_c_cpu)
            eig_vals_c = eig_vals_c.real.to(self._device)
            eig_vecs_c = eig_vecs_c.real.to(self._device)

        eig_vals_c = torch.clamp(eig_vals_c, min=0)
        val_c = torch.diag(torch.sqrt(eig_vals_c))

        # Compute final transfer matrix
        transfer_mat = vec_r @ inv_r @ eig_vecs_c @ val_c @ eig_vecs_c.T @ inv_r @ vec_r.T

        return transfer_mat
