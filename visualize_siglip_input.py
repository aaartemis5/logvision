"""
Visualize what SigLIP actually sees - show the preprocessed image.
"""
import cv2
import numpy as np
import torch
from visionlog.enrichment.siglip_describer import get_siglip_describer
from PIL import Image
import matplotlib.pyplot as plt

print("="*80)
print("VISUALIZING SIGLIP INPUT")
print("="*80)

# Load test image
img = cv2.imread("test_image.jpg")
if img is None:
    print("\n✗ No test_image.jpg found!")
    exit(1)

print(f"\nOriginal image: {img.shape}")

# Get SigLIP describer
desc = get_siglip_describer()
desc._ensure_loaded()

# Convert BGR to RGB
person_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(person_rgb)

# Get preprocessed input (what SigLIP actually sees)
inputs = desc._processor(images=pil_img, return_tensors="pt")
pixel_values = inputs['pixel_values']

print(f"\nPreprocessed shape: {pixel_values.shape}")
print(f"Value range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")

# Convert back to viewable image
# SigLIP preprocessing: normalize to [-1, 1]
# To visualize, we need to denormalize
preprocessed_np = pixel_values.squeeze(0).permute(1, 2, 0).numpy()

# Denormalize from [-1, 1] to [0, 1]
preprocessed_np = (preprocessed_np + 1) / 2
preprocessed_np = np.clip(preprocessed_np, 0, 1)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(person_rgb)
axes[0].set_title(f"Original Image\n{img.shape[1]}x{img.shape[0]}", fontsize=12)
axes[0].axis('off')

# Preprocessed image (what SigLIP sees)
axes[1].imshow(preprocessed_np)
axes[1].set_title(f"What SigLIP Sees\n384x384 (resized & normalized)", fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.savefig("siglip_visualization.png", dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to: siglip_visualization.png")

# Also show color distribution
print("\n" + "="*80)
print("COLOR ANALYSIS")
print("="*80)

# Analyze the preprocessed image colors
r, g, b = preprocessed_np[:,:,0], preprocessed_np[:,:,1], preprocessed_np[:,:,2]

print(f"\nRed channel:   mean={r.mean():.3f}, std={r.std():.3f}")
print(f"Green channel: mean={g.mean():.3f}, std={g.std():.3f}")
print(f"Blue channel:  mean={b.mean():.3f}, std={b.std():.3f}")

# Determine dominant color
if r.mean() > g.mean() and r.mean() > b.mean():
    dominant = "RED"
elif g.mean() > r.mean() and g.mean() > b.mean():
    dominant = "GREEN"
elif b.mean() > r.mean() and b.mean() > g.mean():
    dominant = "BLUE"
else:
    dominant = "NEUTRAL/GRAY"

print(f"\nDominant color channel: {dominant}")

# Check if image is mostly grayscale
color_variance = np.std([r.mean(), g.mean(), b.mean()])
if color_variance < 0.05:
    print("→ Image appears grayscale/neutral (low color variance)")
else:
    print(f"→ Image has color (variance: {color_variance:.3f})")

# Get actual detection result
clothing, emb = desc.describe(img)
print(f"\n" + "="*80)
print(f"SIGLIP DETECTION RESULT: {clothing}")
print("="*80)

print("\nOpen 'siglip_visualization.png' to see what SigLIP is analyzing!")
print("="*80)
