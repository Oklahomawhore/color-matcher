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

from .hist_matcher import HistogramMatcher
from .gpu_utils import get_device, numpy_to_tensor, tensor_to_numpy, gpu_interp


class GPUHistogramMatcher(HistogramMatcher):
    """
    GPU-accelerated histogram matching using PyTorch.
    Supports CUDA, MPS (Apple Silicon), and CPU fallback.
    """

    def __init__(self, *args, **kwargs):
        super(GPUHistogramMatcher, self).__init__(*args, **kwargs)
        self._device = get_device()

    @torch.no_grad()
    def hist_match_gpu(self, src: np.ndarray = None, ref: np.ndarray = None) -> np.ndarray:
        """
        GPU-accelerated channel-wise histogram matching.

        :param src: Source image that requires transfer
        :param ref: Palette image which serves as reference
        :type src: :class:`~numpy:numpy.ndarray`
        :type ref: :class:`~numpy:numpy.ndarray`
        :return: Resulting image after the histogram mapping
        :rtype: np.ndarray
        """

        # override source and reference image with arguments (if provided)
        self._src = src if src is not None else self._src
        self._ref = ref if ref is not None else self._ref

        shape = self._src.shape
        res = np.zeros_like(self._src)

        for ch in range(shape[2]):

            # convert channel data to GPU tensors
            src_vec = numpy_to_tensor(self._src[..., ch].ravel(), self._device)
            ref_vec = numpy_to_tensor(self._ref[..., ch].ravel(), self._device)

            # analyze source histogram on GPU
            src_unique, src_inv, src_counts = torch.unique(
                src_vec, return_inverse=True, return_counts=True
            )
            # analyze reference histogram on GPU
            ref_unique, ref_counts = torch.unique(ref_vec, return_counts=True)

            # compute CDFs on GPU
            src_cdf = torch.cumsum(src_counts.float(), dim=0) / src_vec.numel()
            ref_cdf = torch.cumsum(ref_counts.float(), dim=0) / ref_vec.numel()

            # interpolation on GPU
            interp_vals = gpu_interp(src_cdf, ref_cdf, ref_unique)

            # map and reshape
            result_ch = interp_vals[src_inv].reshape(self._src[..., ch].shape)
            res[..., ch] = tensor_to_numpy(result_ch)

        return res

    @torch.no_grad()
    def hist_match_gpu_cached(self, src: np.ndarray = None, ref_cdf: list = None,
                               ref_vals: list = None) -> np.ndarray:
        """
        GPU histogram matching with pre-computed reference statistics (for video processing).

        :param src: Source image that requires transfer
        :param ref_cdf: Pre-computed reference CDFs per channel (list of tensors)
        :param ref_vals: Pre-computed reference unique values per channel (list of tensors)
        :type src: :class:`~numpy:numpy.ndarray`
        :return: Resulting image after the histogram mapping
        :rtype: np.ndarray
        """

        self._src = src if src is not None else self._src
        shape = self._src.shape
        res = np.zeros_like(self._src)

        for ch in range(shape[2]):
            src_vec = numpy_to_tensor(self._src[..., ch].ravel(), self._device)

            src_unique, src_inv, src_counts = torch.unique(
                src_vec, return_inverse=True, return_counts=True
            )
            src_cdf = torch.cumsum(src_counts.float(), dim=0) / src_vec.numel()

            interp_vals = gpu_interp(src_cdf, ref_cdf[ch], ref_vals[ch])
            result_ch = interp_vals[src_inv].reshape(self._src[..., ch].shape)
            res[..., ch] = tensor_to_numpy(result_ch)

        return res

    @torch.no_grad()
    def precompute_ref_hist(self, ref: np.ndarray) -> tuple:
        """
        Pre-compute reference histogram statistics for video frame reuse.

        :param ref: Reference image
        :type ref: :class:`~numpy:numpy.ndarray`
        :return: Tuple of (ref_cdf_list, ref_vals_list) per channel
        :rtype: tuple
        """
        ref_cdfs = []
        ref_vals = []

        for ch in range(ref.shape[2]):
            ref_vec = numpy_to_tensor(ref[..., ch].ravel(), self._device)
            ref_unique, ref_counts = torch.unique(ref_vec, return_counts=True)
            ref_cdf = torch.cumsum(ref_counts.float(), dim=0) / ref_vec.numel()
            ref_cdfs.append(ref_cdf)
            ref_vals.append(ref_unique)

        return ref_cdfs, ref_vals
