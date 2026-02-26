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

from .reinhard_matcher import ReinhardMatcher, LMS_MAT, LMS_MAT_INV
from .gpu_utils import get_device, numpy_to_tensor, tensor_to_numpy


class GPUReinhardMatcher(ReinhardMatcher):
    """
    GPU-accelerated Reinhard color transfer using PyTorch.
    Supports CUDA, MPS (Apple Silicon), and CPU fallback.
    """

    def __init__(self, *args, **kwargs):
        super(GPUReinhardMatcher, self).__init__(*args, **kwargs)
        self._device = get_device()

        # Pre-compute constant matrices on device
        self._t_lms_mat = torch.tensor(LMS_MAT, dtype=torch.float32, device=self._device)
        self._t_lms_mat_inv = torch.tensor(LMS_MAT_INV, dtype=torch.float32, device=self._device)

        # PCA transform matrices (Rudermann et al.)
        b = np.array([[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(6), 0], [0, 0, 1/np.sqrt(2)]])
        c = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
        self._t_bc = torch.tensor(b @ c, dtype=torch.float32, device=self._device)
        self._t_bc_inv = torch.tensor(c.T @ b, dtype=torch.float32, device=self._device)

        # Cached reference stats for video processing
        self._ref_stats_cached = False
        self._cached_mean_ref = None
        self._cached_std_ref = None

    @torch.no_grad()
    def cache_ref_reinhard_stats(self, ref: np.ndarray):
        """
        Pre-compute and cache reference image Reinhard statistics for video frame reuse.

        :param ref: Reference image
        :type ref: :class:`~numpy:numpy.ndarray`
        """
        m, n, p = ref.shape
        ref_t = numpy_to_tensor(ref.reshape((-1, p)).T, self._device)

        # Replace zeros for numerical stability
        ref_t = torch.where(ref_t == 0, torch.tensor(1.0 / 255.0, device=self._device), ref_t)

        # Convert to LMS and then log-LMS
        lms_ref = torch.mm(self._t_lms_mat, ref_t)
        lms_ref = torch.log10(torch.clamp(lms_ref, min=1e-10))

        # Convert to Lab space
        lab_ref = torch.mm(self._t_bc, lms_ref)

        # Cache statistics
        self._cached_mean_ref = lab_ref.mean(dim=1)
        self._cached_std_ref = lab_ref.std(dim=1)
        self._ref_stats_cached = True

    @torch.no_grad()
    def reinhard_gpu(self, src: np.ndarray = None, ref: np.ndarray = None) -> np.ndarray:
        """
        GPU-accelerated Reinhard color transfer.

        :param src: Source image that requires transfer
        :param ref: Palette image which serves as reference
        :type src: :class:`~numpy:numpy.ndarray`
        :type ref: :class:`~numpy:numpy.ndarray`
        :return: Resulting image after Reinhard color mapping
        :rtype: np.ndarray
        """

        # Override source and reference with arguments (if provided)
        self._src = src if src is not None else self._src
        self._ref = ref if ref is not None else self._ref

        # Get dimensions
        m, n, p = self._src.shape if self.validate_color_chs() else self._src.shape + (1,)

        # Flatten to (C, N) on GPU
        src_t = numpy_to_tensor(self._src.reshape((-1, p)).T, self._device)

        # Replace zeros for numerical stability
        src_t = torch.where(src_t == 0, torch.tensor(1.0 / 255.0, device=self._device), src_t)

        # RGB -> LMS color space
        lms_src = torch.mm(self._t_lms_mat, src_t)

        # LMS -> log-LMS (clamp for numerical stability)
        lms_src = torch.log10(torch.clamp(lms_src, min=1e-10))

        # LMS -> Lab via PCA decorrelation
        lab_src = torch.mm(self._t_bc, lms_src)

        # Compute or use cached reference statistics
        if self._ref_stats_cached:
            mean_ref = self._cached_mean_ref
            std_ref = self._cached_std_ref
        else:
            ref_t = numpy_to_tensor(self._ref.reshape((-1, self._ref.shape[-1])).T, self._device)
            ref_t = torch.where(ref_t == 0, torch.tensor(1.0 / 255.0, device=self._device), ref_t)
            lms_ref = torch.mm(self._t_lms_mat, ref_t)
            lms_ref = torch.log10(torch.clamp(lms_ref, min=1e-10))
            lab_ref = torch.mm(self._t_bc, lms_ref)
            mean_ref = lab_ref.mean(dim=1)
            std_ref = lab_ref.std(dim=1)

        # Source statistics
        mean_src = lab_src.mean(dim=1)
        std_src = lab_src.std(dim=1)

        # Standard deviation ratios
        std_ratios = std_ref / (std_src + 1e-10)

        # Statistical alignment: (x - mean_src) * ratio + mean_ref
        res_lab = ((lab_src.T - mean_src) * std_ratios + mean_ref).T

        # Lab -> log-LMS -> LMS -> RGB
        lms_res = torch.mm(self._t_bc_inv, res_lab)
        lms_res = torch.pow(10.0, lms_res)
        res_t = torch.mm(self._t_lms_mat_inv, lms_res)

        # Convert back to numpy and reshape
        res = tensor_to_numpy(res_t).T.reshape((m, n, p))

        return res
