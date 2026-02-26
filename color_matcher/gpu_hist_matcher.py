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
from .gpu_utils import get_device, numpy_to_tensor, tensor_to_numpy

# 256 bins matches uint8 video precision exactly; higher values add overhead
# without meaningful quality gain for 8-bit-origin frames.
NUM_BINS = 256

# Pre-compute once at module level
_BIN_CENTERS = np.linspace(0.0, 1.0, NUM_BINS)


def _hist_match_channel_np(src_ch: np.ndarray, ref_cdf_np: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """Pure-numpy single-channel histogram match against pre-computed quantized reference CDF."""
    src_vals, src_inv, src_cnts = np.unique(src_ch.ravel(), return_inverse=True, return_counts=True)
    src_cdf = np.cumsum(src_cnts).astype(np.float64) / src_ch.size
    mapped = np.interp(src_cdf, ref_cdf_np, bin_centers)
    return mapped[src_inv].reshape(src_ch.shape)


@torch.no_grad()
def _batch_hist_match_gpu(src: np.ndarray, ref_cdfs: list, device: torch.device,
                          num_bins: int = NUM_BINS) -> np.ndarray:
    """
    Batched GPU histogram matching: all channels in one tensor transfer round-trip.

    Algorithm per channel (all O(n) — no sort):
      1. Quantize float pixels to [0, num_bins-1] integer indices
      2. ``torch.bincount`` → histogram → ``cumsum`` → CDF
      3. ``torch.searchsorted(ref_cdf, src_cdf)`` → bin mapping (B queries into B values)
      4. Build LUT[bin] = matched_bin / (B-1), gather via integer index
    """
    h, w, c = src.shape
    n = h * w

    # single CPU→GPU transfer for all channels
    src_t = torch.from_numpy(src.reshape(n, c).T.astype(np.float32)).to(device)  # (C, N)

    result = torch.empty_like(src_t)

    for ch in range(c):
        vec = src_t[ch]  # (N,)

        # quantize to bin indices
        indices = (vec.clamp(0.0, 1.0) * (num_bins - 1) + 0.5).long().clamp(0, num_bins - 1)

        # build source CDF via bincount (O(n) scatter, no sort)
        counts = torch.bincount(indices, minlength=num_bins).float()
        src_cdf = torch.cumsum(counts, dim=0)
        src_cdf = src_cdf / src_cdf[-1]

        # map each source bin to the closest reference bin via searchsorted (O(B log B), B=256)
        matched = torch.searchsorted(ref_cdfs[ch], src_cdf).clamp(0, num_bins - 1)

        # LUT: bin index → output float
        lut = matched.float() / (num_bins - 1)

        result[ch] = lut[indices]

    # single GPU→CPU transfer
    out = result.cpu().numpy().T.reshape(h, w, c).astype(src.dtype)
    return out


class GPUHistogramMatcher(HistogramMatcher):
    """
    GPU-accelerated histogram matching using PyTorch.
    Supports CUDA, MPS (Apple Silicon), and CPU fallback.

    Uses a **quantized histogram** approach (``torch.bincount`` + LUT) instead
    of ``torch.unique`` so that the hot path is O(n) scatter/gather rather than
    O(n log n) sort – critical for MPS where ``torch.unique`` is pathologically
    slow.

    On MPS specifically, histogram matching uses a numpy fast-path with
    pre-computed reference CDFs, since MPS scatter/sort ops are slower than
    optimised numpy for this workload.
    """

    def __init__(self, *args, **kwargs):
        super(GPUHistogramMatcher, self).__init__(*args, **kwargs)
        self._device = get_device()
        self._use_mps = (self._device.type == 'mps')

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

        self._src = src if src is not None else self._src
        self._ref = ref if ref is not None else self._ref

        if self._use_mps:
            ref_cdfs_np = self._precompute_ref_cdfs_numpy(self._ref)
            return self._match_numpy(self._src, ref_cdfs_np)
        else:
            ref_cdfs = self._precompute_ref_cdfs_torch(self._ref)
            return _batch_hist_match_gpu(self._src, ref_cdfs, self._device)

    @torch.no_grad()
    def hist_match_gpu_cached(self, src: np.ndarray = None, ref_cdf: list = None,
                               ref_vals: list = None) -> np.ndarray:
        """
        GPU histogram matching with pre-computed reference statistics (for video processing).

        :param src: Source image that requires transfer
        :param ref_cdf: Pre-computed reference CDFs per channel.
                        On MPS: list of numpy arrays. On CUDA: list of torch tensors.
        :param ref_vals: Unused (kept for API compatibility). Pass None.
        :type src: :class:`~numpy:numpy.ndarray`
        :return: Resulting image after the histogram mapping
        :rtype: np.ndarray
        """

        self._src = src if src is not None else self._src

        if self._use_mps:
            return self._match_numpy(self._src, ref_cdf)

        return _batch_hist_match_gpu(self._src, ref_cdf, self._device)

    @torch.no_grad()
    def precompute_ref_hist(self, ref: np.ndarray) -> tuple:
        """
        Pre-compute reference histogram CDF for video frame reuse.

        :param ref: Reference image
        :type ref: :class:`~numpy:numpy.ndarray`
        :return: Tuple of (ref_cdf_list, None) – second element kept for API compat
        :rtype: tuple
        """
        if self._use_mps:
            return self._precompute_ref_cdfs_numpy(ref), None
        else:
            return self._precompute_ref_cdfs_torch(ref), None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _precompute_ref_cdfs_torch(self, ref: np.ndarray, num_bins: int = NUM_BINS) -> list:
        """Compute quantized CDF for each channel of *ref* as torch tensors on the CUDA device."""
        ref_cdfs = []
        for ch in range(ref.shape[2]):
            ref_vec = torch.from_numpy(ref[..., ch].ravel().astype(np.float32)).to(self._device)
            indices = (ref_vec.clamp(0.0, 1.0) * (num_bins - 1) + 0.5).long().clamp(0, num_bins - 1)
            counts = torch.bincount(indices, minlength=num_bins).float()
            cdf = torch.cumsum(counts, dim=0)
            cdf = cdf / cdf[-1]
            ref_cdfs.append(cdf)
        return ref_cdfs

    @staticmethod
    def _precompute_ref_cdfs_numpy(ref: np.ndarray, num_bins: int = NUM_BINS) -> list:
        """Compute quantized CDF for each channel of *ref* as numpy arrays (MPS fast-path, zero per-frame overhead)."""
        ref_cdfs_np = []
        for ch in range(ref.shape[2]):
            ref_vec = ref[..., ch].ravel().astype(np.float64)
            indices = np.clip((ref_vec * (num_bins - 1) + 0.5).astype(np.int64), 0, num_bins - 1)
            counts = np.bincount(indices, minlength=num_bins).astype(np.float64)
            cdf = np.cumsum(counts)
            cdf = cdf / cdf[-1]
            ref_cdfs_np.append(cdf)
        return ref_cdfs_np

    @staticmethod
    def _match_numpy(src: np.ndarray, ref_cdfs_np: list, num_bins: int = NUM_BINS) -> np.ndarray:
        """
        Fast numpy-based histogram match using quantized reference CDFs.
        Used as MPS fast-path where torch scatter ops are slow.
        All reference data already in numpy — zero conversion overhead per frame.
        """
        res = np.empty_like(src)
        for ch in range(src.shape[2]):
            res[..., ch] = _hist_match_channel_np(src[..., ch], ref_cdfs_np[ch], _BIN_CENTERS)
        return res
