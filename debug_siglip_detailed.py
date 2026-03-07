"""
Detailed SigLIP debugging - see exactly what the model is detecting and why.
"""
import cv2
import numpy as np
import torch
from visionlog.enrichment.siglip_describer import get_siglip_describer
from PIL import Image

print("="*80)
print("SIGLIP DETAILED DEBUGGING")
print("="*80)

# Load test image
img = cv2.imread("test_image.jpg")
if img is None:
    print("\n✗ No test_image.jpg found!")
    print("  Please provide a test image with people in it.")
    exit(1)

print(f"\nTest image loaded: {img.shape}")

# Get SigLIP describer
desc = get_siglip_describer()

# Force load the model
desc._ensure_loaded()

# Show current prompts
print("\n" + "="*80)
print("CURRENT PROMPTS")
print("="*80)
print("\nColor prompts:")
for i, (label, prompt) in enumerate(zip(desc.COLOUR_LABELS, desc.COLOUR_PROMPTS)):
    print(f"  [{i:2d}] {label:10s} → '{prompt}'")

print("\nGarment prompts:")
for i, (label, prompt) in enumerate(zip(desc.GARMENT_LABELS, desc.GARMENT_PROMPTS)):
    print(f"  [{i:2d}] {label:10s} → '{prompt}'")

# Test on full image
print("\n" + "="*80)
print("TESTING FULL IMAGE")
print("="*80)

person_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(person_rgb)

# Get preprocessed input
inputs = desc._processor(images=pil_img, return_tensors="pt")
print(f"\nPreprocessed image:")
print(f"  Shape: {inputs['pixel_values'].shape}")
print(f"  Min: {inputs['pixel_values'].min():.3f}")
print(f"  Max: {inputs['pixel_values'].max():.3f}")
print(f"  Mean: {inputs['pixel_values'].mean():.3f}")

# Get image embedding
if desc._use_openvino and desc._openvino_model is not None:
    # OpenVINO path
    pixel_values = inputs['pixel_values'].numpy()
    result = desc._openvino_model([pixel_values])
    image_features = torch.from_numpy(result[desc._ov_output])
else:
    # PyTorch path
    with torch.no_grad():
        image_features = desc._model.get_image_features(**inputs)

image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Compute similarities for ALL prompts
colour_sims = (image_features @ desc._colour_embeds.T).squeeze(0)
garment_sims = (image_features @ desc._garment_embeds.T).squeeze(0)

print("\n" + "="*80)
print("COLOR SIMILARITIES (higher = more likely)")
print("="*80)

# Sort by similarity
colour_sorted = sorted(enumerate(colour_sims.tolist()), key=lambda x: x[1], reverse=True)
for rank, (idx, sim) in enumerate(colour_sorted, 1):
    marker = " ← SELECTED" if idx == colour_sims.argmax() else ""
    bar = "█" * int(sim * 50)
    print(f"  {rank:2d}. {desc.COLOUR_LABELS[idx]:10s}: {sim:.4f} {bar}{marker}")

print("\n" + "="*80)
print("GARMENT SIMILARITIES")
print("="*80)

garment_sorted = sorted(enumerate(garment_sims.tolist()), key=lambda x: x[1], reverse=True)
for rank, (idx, sim) in enumerate(garment_sorted, 1):
    marker = " ← SELECTED" if idx == garment_sims.argmax() else ""
    bar = "█" * int(sim * 50)
    print(f"  {rank:2d}. {desc.GARMENT_LABELS[idx]:10s}: {sim:.4f} {bar}{marker}")

# Final result
colour_idx = colour_sims.argmax().item()
garment_idx = garment_sims.argmax().item()
result = f"{desc.COLOUR_LABELS[colour_idx]} {desc.GARMENT_LABELS[garment_idx]}"

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print(f"\n  {result}")

# Check confidence
colour_confidence = colour_sims.max() - colour_sims.sort(descending=True).values[1]
garment_confidence = garment_sims.max() - garment_sims.sort(descending=True).values[1]

print(f"\n  Color confidence: {colour_confidence:.4f}")
print(f"  Garment confidence: {garment_confidence:.4f}")

if colour_confidence < 0.02:
    print("\n  ⚠ WARNING: Color confidence is very low!")
    print("     The model is uncertain about the color.")
    print("     Top 3 colors are too similar.")

if garment_confidence < 0.02:
    print("\n  ⚠ WARNING: Garment confidence is very low!")
    print("     The model is uncertain about the garment type.")

# Analyze actual image colors
print("\n" + "="*80)
print("ACTUAL IMAGE COLOR ANALYSIS")
print("="*80)

# Convert to HSV for color analysis
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

print(f"\nHue (color) distribution:")
print(f"  Mean: {h.mean():.1f}°")
print(f"  Dominant: {np.bincount(h.flatten()).argmax()}°")

# Interpret hue
dominant_hue = np.bincount(h.flatten()).argmax()
if dominant_hue < 10 or dominant_hue > 170:
    color_name = "red"
elif 10 <= dominant_hue < 25:
    color_name = "orange"
elif 25 <= dominant_hue < 40:
    color_name = "yellow"
elif 40 <= dominant_hue < 80:
    color_name = "green"
elif 80 <= dominant_hue < 130:
    color_name = "blue"
elif 130 <= dominant_hue < 150:
    color_name = "purple"
else:
    color_name = "pink/magenta"

print(f"  Interpretation: {color_name}")

print(f"\nSaturation (color intensity):")
print(f"  Mean: {s.mean():.1f}/255")
if s.mean() < 50:
    print(f"  → Low saturation (grayscale/neutral colors)")
else:
    print(f"  → Good saturation (vivid colors)")

print(f"\nValue (brightness):")
print(f"  Mean: {v.mean():.1f}/255")
if v.mean() < 85:
    print(f"  → Dark image")
elif v.mean() > 170:
    print(f"  → Bright image")
else:
    print(f"  → Medium brightness")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if colour_confidence < 0.05:
    print("\n1. Color detection is uncertain - prompts may need adjustment")
    print("   Consider:")
    print("   - More specific color descriptions")
    print("   - Fewer similar colors")
    print("   - Better prompt phrasing")

if s.mean() < 50 and desc.COLOUR_LABELS[colour_idx] not in ["black", "white", "gray"]:
    print("\n2. Image has low saturation but detected vivid color")
    print("   This suggests the model is confused")
    print("   Expected: black, white, or gray")

print("\n" + "="*80)
