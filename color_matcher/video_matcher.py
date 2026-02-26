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
import threading
import queue
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, Optional

from .top_level import ColorMatcher, METHODS
from .io_handler import load_video_meta, create_video_writer, load_img_file
from .normalizer import Normalizer
from .gpu_utils import TORCH_AVAILABLE, HAS_GPU, DEVICE_TYPE

if TORCH_AVAILABLE:
    import torch
    from .gpu_hist_matcher import GPUHistogramMatcher
    from .gpu_mvgd_matcher import GPUTransferMVGD
    from .gpu_reinhard_matcher import GPUReinhardMatcher

try:
    import cv2
except ImportError:
    cv2 = None


class VideoColorMatcher:
    """
    GPU-accelerated video color matching with parallel frame processing.

    Processes video files frame-by-frame using color transfer algorithms,
    with GPU acceleration (CUDA/MPS) and pipelined read/process/write architecture.

    :param src_video: Path to source video file
    :param ref: Reference image (file path or numpy array)
    :param method: Color transfer method (see METHODS)
    :param output_path: Path for output video file (auto-generated if None)
    :param batch_size: Number of frames to process in each batch
    :param num_workers: Number of worker threads for CPU parallel processing
    :param gpu: Enable GPU acceleration (True by default, auto-falls back to CPU)
    :param progress_callback: Optional callback function(current_frame, total_frames)

    :type src_video: str
    :type ref: str or np.ndarray
    :type method: str
    :type output_path: str
    :type batch_size: int
    :type num_workers: int
    :type gpu: bool
    :type progress_callback: callable
    """

    def __init__(self, src_video: str, ref, method: str = 'default',
                 output_path: str = None, batch_size: int = 8, num_workers: int = 4,
                 gpu: bool = True, progress_callback: Optional[Callable] = None):

        if cv2 is None:
            raise ImportError('OpenCV is required for video processing. Install with: pip install opencv-python')

        self._src_video = src_video
        self._method = method.lower() if method else 'default'
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._use_gpu = gpu and HAS_GPU and TORCH_AVAILABLE
        self._progress_callback = progress_callback

        # Load reference image
        if isinstance(ref, str):
            self._ref = load_img_file(ref)
        elif isinstance(ref, np.ndarray):
            self._ref = ref
        else:
            raise TypeError('ref must be a file path string or numpy array')

        # Ensure ref has 3 dimensions
        if self._ref.ndim == 2:
            self._ref = self._ref[..., np.newaxis]

        # Load video metadata
        self._meta = load_video_meta(src_video)

        # Auto-generate output path
        if output_path is None:
            base, ext = os.path.splitext(src_video)
            ext = ext if ext else '.mp4'
            self._output_path = '%s_%s%s' % (base, method, ext)
        else:
            self._output_path = output_path

        # Pre-compute reference statistics for GPU path
        self._ref_hist_cache = None
        self._gpu_matchers = {}

        if self._use_gpu:
            self._init_gpu_cache()

    def _init_gpu_cache(self):
        """Pre-compute reference statistics on GPU for reuse across frames."""

        # Create GPU matchers and cache reference stats
        if self._method in ('hm', 'hm-mvgd-hm', 'hm-mkl-hm'):
            gpu_hist = GPUHistogramMatcher(src=self._ref, ref=self._ref, method=self._method)
            ref_cdfs, ref_vals = gpu_hist.precompute_ref_hist(self._ref)
            self._ref_hist_cache = (ref_cdfs, ref_vals)
            self._gpu_matchers['hist'] = gpu_hist

        if self._method in ('default', 'mvgd', 'mkl', 'hm-mvgd-hm', 'hm-mkl-hm'):
            gpu_mvgd = GPUTransferMVGD(src=self._ref, ref=self._ref, method=self._method)
            gpu_mvgd.cache_ref_stats(self._ref)
            self._gpu_matchers['mvgd'] = gpu_mvgd

        if self._method == 'reinhard':
            gpu_reinhard = GPUReinhardMatcher(src=self._ref, ref=self._ref, method=self._method)
            gpu_reinhard.cache_ref_reinhard_stats(self._ref)
            self._gpu_matchers['reinhard'] = gpu_reinhard

    def _process_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame using GPU-accelerated color transfer with cached reference stats.

        :param frame: Input video frame (H, W, C)
        :type frame: np.ndarray
        :return: Color-transferred frame
        :rtype: np.ndarray
        """

        # Ensure 3D
        if frame.ndim == 2:
            frame = frame[..., np.newaxis]

        result = frame.copy()

        if self._method == 'hm':
            matcher = self._gpu_matchers['hist']
            result = matcher.hist_match_gpu_cached(
                src=result, ref_cdf=self._ref_hist_cache[0], ref_vals=self._ref_hist_cache[1]
            )

        elif self._method == 'reinhard':
            matcher = self._gpu_matchers['reinhard']
            result = matcher.reinhard_gpu(src=result, ref=self._ref)

        elif self._method in ('default', 'mvgd', 'mkl'):
            matcher = self._gpu_matchers['mvgd']
            result = matcher.multivar_transfer_gpu(src=result, ref=self._ref)

        elif self._method in ('hm-mvgd-hm', 'hm-mkl-hm'):
            hist_matcher = self._gpu_matchers['hist']
            mvgd_matcher = self._gpu_matchers['mvgd']

            result = hist_matcher.hist_match_gpu_cached(
                src=result, ref_cdf=self._ref_hist_cache[0], ref_vals=self._ref_hist_cache[1]
            )
            result = mvgd_matcher.multivar_transfer_gpu(src=result, ref=self._ref)
            result = hist_matcher.hist_match_gpu_cached(
                src=result, ref_cdf=self._ref_hist_cache[0], ref_vals=self._ref_hist_cache[1]
            )

        return result

    def _process_frame_cpu(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame using CPU-based color transfer.

        :param frame: Input video frame (H, W, C)
        :type frame: np.ndarray
        :return: Color-transferred frame
        :rtype: np.ndarray
        """

        if frame.ndim == 2:
            frame = frame[..., np.newaxis]

        result = ColorMatcher(src=frame, ref=self._ref, method=self._method, gpu=False).main()
        return result

    def process(self) -> str:
        """
        Process the entire video with pipelined read/process/write architecture.

        On GPU (CUDA/MPS): Three-thread pipeline with GPU-accelerated frame processing.
        On CPU: Multi-process parallel frame processing.

        :return: Path to the output video file
        :rtype: str
        """

        if self._use_gpu:
            return self._process_gpu_pipeline()
        else:
            return self._process_cpu_parallel()

    def _process_gpu_pipeline(self) -> str:
        """
        GPU pipeline: Reader thread -> GPU processing -> Writer thread.
        MPS (Apple Silicon): Single GPU stream with threaded I/O.
        CUDA: Multiple CUDA streams for frame-level parallelism.
        """

        frame_queue = queue.Queue(maxsize=self._batch_size * 2)
        result_queue = queue.Queue(maxsize=self._batch_size * 2)
        total_frames = self._meta['frame_count']
        fps = self._meta['fps']
        width = self._meta['width']
        height = self._meta['height']

        # Sentinel value for end of stream
        SENTINEL = None
        error_holder = [None]  # mutable container for thread error propagation

        def reader_thread():
            """Read frames from video into queue."""
            try:
                cap = cv2.VideoCapture(self._src_video)
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # OpenCV reads BGR, convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_queue.put((frame_idx, frame_rgb))
                    frame_idx += 1
                cap.release()
            except Exception as e:
                error_holder[0] = e
            finally:
                frame_queue.put(SENTINEL)

        def processor_thread():
            """Process frames from queue using GPU."""
            try:
                while True:
                    item = frame_queue.get()
                    if item is SENTINEL:
                        break
                    frame_idx, frame = item

                    # Normalize frame to float for processing
                    src_dtype = frame.dtype
                    frame_float = Normalizer(frame).type_norm(new_min=0.0, new_max=1.0) if np.issubdtype(src_dtype, np.integer) else frame.astype(np.float64)

                    # Ensure 3 color channels
                    if frame_float.ndim == 2:
                        frame_float = frame_float[..., np.newaxis]

                    # GPU processing
                    result = self._process_frame_gpu(frame_float)

                    # Normalize back to uint8
                    result = Normalizer(result).uint8_norm()

                    result_queue.put((frame_idx, result))

                    # Report progress
                    if self._progress_callback:
                        self._progress_callback(frame_idx + 1, total_frames)
            except Exception as e:
                error_holder[0] = e
            finally:
                result_queue.put(SENTINEL)

        def writer_thread():
            """Write processed frames to output video."""
            try:
                writer = create_video_writer(self._output_path, fps, width, height)
                while True:
                    item = result_queue.get()
                    if item is SENTINEL:
                        break
                    frame_idx, result = item

                    # Convert RGB back to BGR for OpenCV
                    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    writer.write(result_bgr)
                writer.release()
            except Exception as e:
                error_holder[0] = e

        # Launch pipeline threads
        t_reader = threading.Thread(target=reader_thread, name='VideoReader', daemon=True)
        t_processor = threading.Thread(target=processor_thread, name='GPUProcessor', daemon=True)
        t_writer = threading.Thread(target=writer_thread, name='VideoWriter', daemon=True)

        t_reader.start()
        t_processor.start()
        t_writer.start()

        # Wait for completion
        t_reader.join()
        t_processor.join()
        t_writer.join()

        # Check for errors
        if error_holder[0] is not None:
            raise RuntimeError('Video processing failed: %s' % str(error_holder[0]))

        return self._output_path

    def _process_cpu_parallel(self) -> str:
        """
        CPU parallel processing using ProcessPoolExecutor for frame-level parallelism.
        """

        cap = cv2.VideoCapture(self._src_video)
        total_frames = self._meta['frame_count']
        fps = self._meta['fps']
        width = self._meta['width']
        height = self._meta['height']

        writer = create_video_writer(self._output_path, fps, width, height)

        # Read all frames first (for ordered output)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        # Process frames in batches using thread pool
        # (Using ThreadPool instead of ProcessPool to avoid serialization overhead of numpy arrays)
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            for batch_start in range(0, len(frames), self._batch_size):
                batch_end = min(batch_start + self._batch_size, len(frames))
                batch = frames[batch_start:batch_end]

                def process_one(frame):
                    src_dtype = frame.dtype
                    frame_float = Normalizer(frame).type_norm(new_min=0.0, new_max=1.0) if np.issubdtype(src_dtype, np.integer) else frame.astype(np.float64)
                    if frame_float.ndim == 2:
                        frame_float = frame_float[..., np.newaxis]
                    result = self._process_frame_cpu(frame_float)
                    return Normalizer(result).uint8_norm()

                results = list(executor.map(process_one, batch))

                for i, result in enumerate(results):
                    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    writer.write(result_bgr)

                    if self._progress_callback:
                        self._progress_callback(batch_start + i + 1, total_frames)

        writer.release()
        return self._output_path

    @property
    def meta(self) -> dict:
        """Video metadata dictionary."""
        return self._meta

    @property
    def output_path(self) -> str:
        """Output video file path."""
        return self._output_path
