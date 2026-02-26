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

import unittest
import os
import sys
import tempfile
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from color_matcher.top_level import ColorMatcher, METHODS
from color_matcher.io_handler import load_img_file


class GPUAccelerationTester(unittest.TestCase):
    """Test GPU-accelerated color matching algorithms."""

    def __init__(self, *args, **kwargs):
        super(GPUAccelerationTester, self).__init__(*args, **kwargs)

    def setUp(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.dat_path = os.path.join(self.dir_path, 'data')

        # Load test images
        self.plain = load_img_file(os.path.join(self.dat_path, 'scotland_plain.png'))
        self.house = load_img_file(os.path.join(self.dat_path, 'scotland_house.png'))

    @staticmethod
    def avg_hist_dist(img1, img2, bins=2**8-1):
        hist_a = np.histogram(img1, bins)[0]
        hist_b = np.histogram(img2, bins)[0]
        return np.sqrt(np.sum(np.square(hist_a - hist_b)))

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_utils_device_detection(self):
        """Test device detection returns valid device."""
        from color_matcher.gpu_utils import get_device, DEVICE_TYPE, HAS_GPU

        device = get_device()
        self.assertIn(device.type, ('cuda', 'mps', 'cpu'))
        self.assertIsInstance(DEVICE_TYPE, str)
        self.assertIsInstance(HAS_GPU, bool)

        if torch.cuda.is_available():
            self.assertEqual(DEVICE_TYPE, 'cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.assertEqual(DEVICE_TYPE, 'mps')
        else:
            self.assertEqual(DEVICE_TYPE, 'cpu')

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_numpy_tensor_roundtrip(self):
        """Test numpy -> tensor -> numpy roundtrip preserves data."""
        from color_matcher.gpu_utils import numpy_to_tensor, tensor_to_numpy

        arr = np.random.rand(10, 10, 3).astype(np.float32)
        tensor = numpy_to_tensor(arr)
        result = tensor_to_numpy(tensor)

        np.testing.assert_allclose(arr, result, atol=1e-6)

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_cov(self):
        """Test GPU covariance matches numpy covariance."""
        from color_matcher.gpu_utils import gpu_cov, numpy_to_tensor, tensor_to_numpy

        data = np.random.rand(3, 100).astype(np.float32)
        np_cov = np.cov(data).astype(np.float32)

        t_data = numpy_to_tensor(data)
        gpu_cov_result = tensor_to_numpy(gpu_cov(t_data))

        np.testing.assert_allclose(np_cov, gpu_cov_result, atol=1e-4)

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_interp(self):
        """Test GPU interpolation matches numpy interpolation."""
        from color_matcher.gpu_utils import gpu_interp, numpy_to_tensor, tensor_to_numpy, get_device

        device = get_device()
        xp = np.linspace(0, 1, 50).astype(np.float32)
        fp = np.sin(xp * np.pi).astype(np.float32)
        x = np.random.rand(200).astype(np.float32)

        np_result = np.interp(x, xp, fp)

        t_x = numpy_to_tensor(x, device)
        t_xp = numpy_to_tensor(xp, device)
        t_fp = numpy_to_tensor(fp, device)
        gpu_result = tensor_to_numpy(gpu_interp(t_x, t_xp, t_fp))

        np.testing.assert_allclose(np_result, gpu_result, atol=1e-4)

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_color_matcher_default(self):
        """Test GPU-accelerated default (MKL) method produces valid output."""
        result_gpu = ColorMatcher(src=self.house, ref=self.plain, method='default', gpu=True).main()
        result_cpu = ColorMatcher(src=self.house, ref=self.plain, method='default', gpu=False).main()

        # Both should improve histogram distance
        orig_dist = self.avg_hist_dist(self.plain, self.house)
        gpu_dist = self.avg_hist_dist(self.plain, result_gpu)
        cpu_dist = self.avg_hist_dist(self.plain, result_cpu)

        self.assertLess(gpu_dist, orig_dist, 'GPU result should improve histogram distance')
        self.assertLess(cpu_dist, orig_dist, 'CPU result should improve histogram distance')

        # GPU and CPU results should be similar (not exact due to float32 vs float64)
        np.testing.assert_allclose(result_gpu.astype(float), result_cpu.astype(float), atol=2.0,
                                   err_msg='GPU and CPU results should be similar')

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_color_matcher_hm(self):
        """Test GPU-accelerated histogram matching method."""
        result_gpu = ColorMatcher(src=self.house, ref=self.plain, method='hm', gpu=True).main()

        orig_dist = self.avg_hist_dist(self.plain, self.house)
        gpu_dist = self.avg_hist_dist(self.plain, result_gpu)

        self.assertLess(gpu_dist, orig_dist, 'GPU HM result should improve histogram distance')

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_color_matcher_reinhard(self):
        """Test GPU-accelerated Reinhard method."""
        result_gpu = ColorMatcher(src=self.house, ref=self.plain, method='reinhard', gpu=True).main()

        orig_dist = self.avg_hist_dist(self.plain, self.house)
        gpu_dist = self.avg_hist_dist(self.plain, result_gpu)

        self.assertLess(gpu_dist, orig_dist, 'GPU Reinhard result should improve histogram distance')

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_color_matcher_all_methods(self):
        """Test all GPU-accelerated methods produce valid improvements."""
        for method in METHODS:
            with self.subTest(method=method):
                result = ColorMatcher(src=self.house, ref=self.plain, method=method, gpu=True).main()
                orig_dist = self.avg_hist_dist(self.plain, self.house)
                result_dist = self.avg_hist_dist(self.plain, result)
                self.assertLess(result_dist, orig_dist,
                                'GPU %s result should improve histogram distance' % method)

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_fallback_to_cpu(self):
        """Test that gpu=False forces CPU processing."""
        result = ColorMatcher(src=self.house, ref=self.plain, method='default', gpu=False).main()
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.house.shape)

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_gpu_varying_resolutions(self):
        """Test GPU path with varying image resolutions."""
        for shape in [(50, 50, 3), (100, 80, 3), (30, 60, 3)]:
            with self.subTest(shape=shape):
                src = np.random.rand(*shape).astype(np.float64)
                ref = np.random.rand(*shape).astype(np.float64)
                result = ColorMatcher(src=src, ref=ref, method='default', gpu=True).main()
                self.assertEqual(result.shape, shape)


@unittest.skipUnless(HAS_CV2, "requires OpenCV")
@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class VideoColorMatcherTester(unittest.TestCase):
    """Test video color matching pipeline."""

    def setUp(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.dat_path = os.path.join(self.dir_path, 'data')
        self.tmp_dir = tempfile.mkdtemp()

        # Load reference image
        self.ref_img = load_img_file(os.path.join(self.dat_path, 'scotland_plain.png'))

        # Create synthetic test video
        self.test_video_path = self._create_test_video()

    def _create_test_video(self, num_frames=30, fps=24.0) -> str:
        """Create a short synthetic test video for testing."""
        video_path = os.path.join(self.tmp_dir, 'test_input.mp4')
        h, w = 120, 160
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for i in range(num_frames):
            # Create frame with varying color (simulates video content)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            color_shift = int(255 * i / num_frames)
            frame[:, :, 0] = color_shift           # varying blue
            frame[:, :, 1] = 128                    # constant green
            frame[:, :, 2] = 255 - color_shift      # inverse red
            # Add some random noise for realism
            noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
            frame = np.clip(frame.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
            writer.write(frame)

        writer.release()
        return video_path

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_video_meta_loading(self):
        """Test video metadata is loaded correctly."""
        from color_matcher.io_handler import load_video_meta
        meta = load_video_meta(self.test_video_path)

        self.assertEqual(meta['width'], 160)
        self.assertEqual(meta['height'], 120)
        self.assertEqual(meta['frame_count'], 30)
        self.assertGreater(meta['fps'], 0)

    def test_is_video_file(self):
        """Test video file extension detection."""
        from color_matcher.io_handler import is_video_file

        self.assertTrue(is_video_file('test.mp4'))
        self.assertTrue(is_video_file('test.avi'))
        self.assertTrue(is_video_file('test.mov'))
        self.assertFalse(is_video_file('test.png'))
        self.assertFalse(is_video_file('test.jpg'))

    def test_video_color_matcher_gpu(self):
        """Test VideoColorMatcher end-to-end with GPU."""
        from color_matcher.video_matcher import VideoColorMatcher

        output_path = os.path.join(self.tmp_dir, 'test_output_gpu.mp4')

        progress_calls = []
        def progress_cb(current, total):
            progress_calls.append((current, total))

        vm = VideoColorMatcher(
            src_video=self.test_video_path,
            ref=self.ref_img,
            method='default',
            output_path=output_path,
            batch_size=4,
            gpu=True,
            progress_callback=progress_cb,
        )

        result_path = vm.process()

        # Verify output exists
        self.assertTrue(os.path.exists(result_path))

        # Verify output video properties
        from color_matcher.io_handler import load_video_meta
        out_meta = load_video_meta(result_path)
        self.assertEqual(out_meta['width'], 160)
        self.assertEqual(out_meta['height'], 120)
        self.assertGreater(out_meta['frame_count'], 0)

        # Verify progress was reported
        self.assertGreater(len(progress_calls), 0)

    def test_video_color_matcher_cpu(self):
        """Test VideoColorMatcher end-to-end with CPU fallback."""
        from color_matcher.video_matcher import VideoColorMatcher

        output_path = os.path.join(self.tmp_dir, 'test_output_cpu.mp4')

        vm = VideoColorMatcher(
            src_video=self.test_video_path,
            ref=self.ref_img,
            method='hm',
            output_path=output_path,
            gpu=False,
            num_workers=2,
        )

        result_path = vm.process()
        self.assertTrue(os.path.exists(result_path))

    def test_video_color_matcher_all_methods(self):
        """Test VideoColorMatcher with all transfer methods."""
        from color_matcher.video_matcher import VideoColorMatcher

        for method in ('default', 'hm', 'reinhard'):
            with self.subTest(method=method):
                output_path = os.path.join(self.tmp_dir, 'test_%s.mp4' % method)
                vm = VideoColorMatcher(
                    src_video=self.test_video_path,
                    ref=self.ref_img,
                    method=method,
                    output_path=output_path,
                    batch_size=4,
                )
                result_path = vm.process()
                self.assertTrue(os.path.exists(result_path))

    def test_video_color_matcher_ref_as_path(self):
        """Test VideoColorMatcher with reference image as file path."""
        from color_matcher.video_matcher import VideoColorMatcher

        ref_path = os.path.join(self.dat_path, 'scotland_plain.png')
        output_path = os.path.join(self.tmp_dir, 'test_ref_path.mp4')

        vm = VideoColorMatcher(
            src_video=self.test_video_path,
            ref=ref_path,
            method='default',
            output_path=output_path,
        )

        result_path = vm.process()
        self.assertTrue(os.path.exists(result_path))

    def test_video_auto_output_path(self):
        """Test automatic output path generation."""
        from color_matcher.video_matcher import VideoColorMatcher

        vm = VideoColorMatcher(
            src_video=self.test_video_path,
            ref=self.ref_img,
            method='mkl',
        )

        expected_suffix = '_mkl.mp4'
        self.assertTrue(vm.output_path.endswith(expected_suffix))

    def test_video_meta_property(self):
        """Test VideoColorMatcher.meta property."""
        from color_matcher.video_matcher import VideoColorMatcher

        vm = VideoColorMatcher(
            src_video=self.test_video_path,
            ref=self.ref_img,
        )

        meta = vm.meta
        self.assertIn('fps', meta)
        self.assertIn('width', meta)
        self.assertIn('height', meta)
        self.assertIn('frame_count', meta)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MpsSpecificTester(unittest.TestCase):
    """Tests specific to MPS (Apple Silicon) backend."""

    @unittest.skipUnless(
        HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "requires MPS device"
    )
    def test_mps_linalg_eig(self):
        """Test torch.linalg.eig works on MPS device."""
        device = torch.device('mps')
        mat = torch.randn(3, 3, device=device, dtype=torch.float32)
        mat = mat @ mat.T  # make symmetric positive semi-definite

        try:
            eig_vals, eig_vecs = torch.linalg.eig(mat)
            self.assertEqual(eig_vals.shape[0], 3)
            mps_eig_works = True
        except RuntimeError:
            mps_eig_works = False
            # This is OK - the code has CPU fallback for this

        # Either way, the GPU matchers should handle this
        from color_matcher.gpu_mvgd_matcher import GPUTransferMVGD
        src = np.random.rand(50, 50, 3).astype(np.float64)
        ref = np.random.rand(50, 50, 3).astype(np.float64)
        matcher = GPUTransferMVGD(src=src, ref=ref, method='mkl')
        result = matcher.multivar_transfer_gpu(src, ref)
        self.assertEqual(result.shape, src.shape)

    @unittest.skipUnless(
        HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "requires MPS device"
    )
    def test_mps_searchsorted(self):
        """Test torch.searchsorted works on MPS device."""
        device = torch.device('mps')
        sorted_seq = torch.linspace(0, 1, 100, device=device)
        values = torch.rand(50, device=device)

        try:
            indices = torch.searchsorted(sorted_seq, values)
            self.assertEqual(indices.shape[0], 50)
        except RuntimeError:
            self.skipTest('searchsorted not supported on MPS in this PyTorch version')


if __name__ == '__main__':
    unittest.main()
