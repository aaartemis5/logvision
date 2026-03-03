"""
Performance comparison: PyTorch vs OpenVINO for SigLIP

This script benchmarks the SigLIP model with both PyTorch and OpenVINO backends
to demonstrate the performance improvements from OpenVINO optimization.
"""
import time
import cv2
import numpy as np
from pathlib import Path

print("="*70)
print("SigLIP Performance Comparison: PyTorch vs OpenVINO")
print("="*70)

# Load test image
test_img = cv2.imread("test_image.jpg")
if test_img is None:
    print("Creating dummy test image...")
    test_img = np.random.randint(0, 255, (480, 320, 3), dtype=np.uint8)

print(f"\nTest image: {test_img.shape}")

# Test 1: PyTorch Backend
print("\n" + "="*70)
print("TEST 1: PyTorch Backend (Baseline)")
print("="*70)

from visionlog.enrichment.siglip_describer import SigLIPDescriber

pytorch_desc = SigLIPDescriber(use_openvino=False)

# Warm-up
print("Warming up PyTorch model...")
for _ in range(3):
    pytorch_desc.describe(test_img)

# Benchmark
print("Running 20 inferences...")
pytorch_times = []
for i in range(20):
    start = time.time()
    clothing, emb = pytorch_desc.describe(test_img)
    elapsed = (time.time() - start) * 1000
    pytorch_times.append(elapsed)
    if i == 0:
        print(f"  First result: {clothing}")

pytorch_avg = np.mean(pytorch_times)
pytorch_std = np.std(pytorch_times)
pytorch_min = np.min(pytorch_times)
pytorch_max = np.max(pytorch_times)

print(f"\nPyTorch Results:")
print(f"  Average: {pytorch_avg:.1f} ms")
print(f"  Std Dev: {pytorch_std:.1f} ms")
print(f"  Min:     {pytorch_min:.1f} ms")
print(f"  Max:     {pytorch_max:.1f} ms")
print(f"  FPS:     {1000/pytorch_avg:.2f}")

# Test 2: OpenVINO Backend
print("\n" + "="*70)
print("TEST 2: OpenVINO Backend (Optimized)")
print("="*70)

openvino_desc = SigLIPDescriber(use_openvino=True)

# Warm-up
print("Warming up OpenVINO model...")
for _ in range(3):
    openvino_desc.describe(test_img)

# Benchmark
print("Running 20 inferences...")
openvino_times = []
for i in range(20):
    start = time.time()
    clothing, emb = openvino_desc.describe(test_img)
    elapsed = (time.time() - start) * 1000
    openvino_times.append(elapsed)
    if i == 0:
        print(f"  First result: {clothing}")

openvino_avg = np.mean(openvino_times)
openvino_std = np.std(openvino_times)
openvino_min = np.min(openvino_times)
openvino_max = np.max(openvino_times)

print(f"\nOpenVINO Results:")
print(f"  Average: {openvino_avg:.1f} ms")
print(f"  Std Dev: {openvino_std:.1f} ms")
print(f"  Min:     {openvino_min:.1f} ms")
print(f"  Max:     {openvino_max:.1f} ms")
print(f"  FPS:     {1000/openvino_avg:.2f}")

# Comparison
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)

speedup = pytorch_avg / openvino_avg
improvement = ((pytorch_avg - openvino_avg) / pytorch_avg) * 100

print(f"\nSpeedup:     {speedup:.2f}x faster")
print(f"Improvement: {improvement:.1f}% reduction in latency")
print(f"Time saved:  {pytorch_avg - openvino_avg:.1f} ms per inference")

# Memory comparison
print("\n" + "="*70)
print("MEMORY USAGE")
print("="*70)

ov_model_dir = Path("models/siglip_openvino")
if ov_model_dir.exists():
    xml_size = (ov_model_dir / "siglip_vision.xml").stat().st_size / 1024 / 1024
    bin_size = (ov_model_dir / "siglip_vision.bin").stat().st_size / 1024 / 1024
    print(f"\nOpenVINO model size: {xml_size + bin_size:.1f} MB")
    print(f"  XML (graph): {xml_size:.1f} MB")
    print(f"  BIN (weights): {bin_size:.1f} MB")

# Final verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)

if speedup >= 1.5:
    print(f"\n✓✓✓ OpenVINO provides {speedup:.2f}x speedup - EXCELLENT!")
    print("    Recommended for production use.")
elif speedup >= 1.2:
    print(f"\n✓✓ OpenVINO provides {speedup:.2f}x speedup - GOOD!")
    print("   Recommended for CPU-bound deployments.")
elif speedup >= 1.0:
    print(f"\n✓ OpenVINO provides {speedup:.2f}x speedup - MARGINAL")
    print("  Consider using for consistency, but gains are small.")
else:
    print(f"\n✗ OpenVINO is slower ({speedup:.2f}x)")
    print("  Stick with PyTorch for this hardware.")

print("\n" + "="*70)
