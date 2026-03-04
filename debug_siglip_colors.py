"""
Debug script to see what colors SigLIP is detecting and why.
"""
import cv2
import numpy as np
from visionlog.enrichment.siglip_describer import get_siglip_describer

# Load test image
img = cv2.imread("test_image.jpg")
if img is None:
    print("No test image found, creating dummy")
    img = np.random.randint(0, 255, (480, 320, 3), dtype=np.uint8)

print("Testing SigLIP color detection...")
print("="*70)

desc = get_siglip_describer()

# Test on full image
clothing, emb = desc.describe(img)
print(f"\nFull image result: {clothing}")

# Show what the processor is doing
from PIL import Image
person_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(person_rgb)

# Get the actual pixel values being sent to the model
inputs = desc._processor(images=pil_img, return_tensors="pt")
pixel_values = inputs['pixel_values']

print(f"\nPreprocessed image stats:")
print(f"  Shape: {pixel_values.shape}")
print(f"  Min: {pixel_values.min():.3f}")
print(f"  Max: {pixel_values.max():.3f}")
print(f"  Mean: {pixel_values.mean():.3f}")
print(f"  Std: {pixel_values.std():.3f}")

# Get image embedding
import torch
with torch.no_grad():
    image_features = desc._model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Check similarity with ALL color prompts
print(f"\n{'='*70}")
print("Color Similarities (higher = more likely):")
print(f"{'='*70}")

colour_sims = (image_features @ desc._colour_embeds.T).squeeze(0)
for i, (label, sim) in enumerate(zip(desc.COLOUR_LABELS, colour_sims)):
    marker = " ← SELECTED" if i == colour_sims.argmax() else ""
    print(f"  {label:12s}: {sim:.4f}{marker}")

print(f"\n{'='*70}")
print("Garment Similarities:")
print(f"{'='*70}")

garment_sims = (image_features @ desc._garment_embeds.T).squeeze(0)
for i, (label, sim) in enumerate(zip(desc.GARMENT_LABELS, garment_sims)):
    marker = " ← SELECTED" if i == garment_sims.argmax() else ""
    print(f"  {label:12s}: {sim:.4f}{marker}")

print(f"\n{'='*70}")
print("\nDiagnosis:")
print(f"{'='*70}")

# Check if similarities are too close
colour_diff = colour_sims.max() - colour_sims.sort(descending=True).values[1]
garment_diff = garment_sims.max() - garment_sims.sort(descending=True).values[1]

print(f"\nColor confidence: {colour_diff:.4f}")
print(f"Garment confidence: {garment_diff:.4f}")

if colour_diff < 0.05:
    print("\n⚠ WARNING: Color similarities are too close!")
    print("   The model is uncertain about the color.")
    print("   This suggests the prompts may need improvement.")

if garment_diff < 0.05:
    print("\n⚠ WARNING: Garment similarities are too close!")
    print("   The model is uncertain about the garment type.")
    print("   This suggests the prompts may need improvement.")

# Show actual image colors
print(f"\n{'='*70}")
print("Actual Image Color Distribution:")
print(f"{'='*70}")

# Convert to HSV to analyze colors
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

print(f"\nHue (color):")
print(f"  Mean: {h.mean():.1f} (0-179 scale)")
print(f"  Dominant hue: {np.bincount(h.flatten()).argmax()}")

print(f"\nSaturation (color intensity):")
print(f"  Mean: {s.mean():.1f} (0-255 scale)")
print(f"  Low saturation (<50): {(s < 50).sum() / s.size * 100:.1f}%")

print(f"\nValue (brightness):")
print(f"  Mean: {v.mean():.1f} (0-255 scale)")

# Interpret
if s.mean() < 50:
    print("\n💡 Image has low saturation - likely grayscale/neutral colors")
    print("   Expected: black, white, grey, navy")
else:
    print("\n💡 Image has good saturation - colors should be detectable")

print(f"\n{'='*70}")
