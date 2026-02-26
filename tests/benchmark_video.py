#!/usr/bin/env python3
"""
Quick test: GPU vs CPU video color matching on a short clip.
Source: tests/data/bilibili_BV1T8fABiEsi.mp4 (first 300 frames)
Reference: tests/data/tiger_colormatch.png
"""

import time
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_matcher.video_matcher import VideoColorMatcher
from color_matcher.gpu_utils import DEVICE_TYPE, HAS_GPU
from color_matcher.io_handler import load_video_meta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SRC_VIDEO = os.path.join(DATA_DIR, 'bilibili_BV1T8fABiEsi.mp4')
REF_IMAGE = os.path.join(DATA_DIR, 'tiger_colormatch.png')
MAX_FRAMES = 300


def create_short_clip(src_path, out_path, max_frames):
    """Extract first N frames into a short test clip."""
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    return count, fps, w, h


def progress(cur, total):
    if cur % 50 == 0 or cur == total:
        pct = cur * 100 // total
        print(f'\r  Frame {cur}/{total} ({pct}%)', end='', flush=True)


def run_test(video_path, method, gpu):
    tag = 'GPU-MPS' if gpu else 'CPU'
    output = os.path.join(DATA_DIR, f'output_{"gpu" if gpu else "cpu"}_{method}.mp4')

    vm = VideoColorMatcher(
        src_video=video_path,
        ref=REF_IMAGE,
        method=method,
        output_path=output,
        batch_size=8,
        gpu=gpu,
        num_workers=4,
        progress_callback=progress,
    )

    total = vm.meta['frame_count']
    print(f'  {tag} / {method}: {total} frames ...', end='')

    start = time.time()
    vm.process()
    elapsed = time.time() - start
    fps_proc = total / elapsed

    sz = os.path.getsize(output) / (1024 * 1024)
    print(f'\r  {tag:8s} / {method:12s}: {elapsed:6.1f}s  ({fps_proc:5.1f} fps)  [{sz:.1f} MB]')

    return elapsed, fps_proc


if __name__ == '__main__':
    print(f'Device: {DEVICE_TYPE} (GPU: {HAS_GPU})')
    print(f'Extracting first {MAX_FRAMES} frames from source video...')

    clip_path = os.path.join(DATA_DIR, 'test_clip_300f.mp4')
    nframes, fps, w, h = create_short_clip(SRC_VIDEO, clip_path, MAX_FRAMES)
    print(f'Test clip: {nframes} frames, {w}x{h} @{fps:.1f}fps\n')

    results = {}
    methods = ['default', 'hm', 'reinhard']

    for method in methods:
        if HAS_GPU:
            t, f = run_test(clip_path, method, gpu=True)
            results[f'GPU-{method}'] = (t, f)

        t, f = run_test(clip_path, method, gpu=False)
        results[f'CPU-{method}'] = (t, f)

    print(f'\n{"="*60}')
    print(f'BENCHMARK SUMMARY ({nframes} frames, {w}x{h})')
    print(f'{"="*60}')
    for name, (t, f) in results.items():
        print(f'  {name:20s}: {t:7.1f}s  ({f:5.1f} fps)')

    for method in methods:
        gpu_key = f'GPU-{method}'
        cpu_key = f'CPU-{method}'
        if gpu_key in results and cpu_key in results:
            speedup = results[cpu_key][0] / results[gpu_key][0]
            print(f'  {method:20s}  GPU speedup: {speedup:.2f}x')
