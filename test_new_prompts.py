"""Test the new SigLIP prompts on real images"""
import cv2
from visionlog.enrichment.siglip_describer import get_siglip_describer

print("="*70)
print("Testing New SigLIP Prompts")
print("="*70)

desc = get_siglip_describer()

print(f"\nNew color labels: {desc.COLOUR_LABELS}")
print(f"New garment labels: {desc.GARMENT_LABELS}")

# Test on real image
img = cv2.imread('test_image.jpg')
if img is not None:
    print(f"\n{'='*70}")
    print("Testing on test_image.jpg")
    print(f"{'='*70}")
    
    # Test full image
    clothing, emb = desc.describe(img)
    print(f"\nFull image: {clothing}")
    
    # Test different regions
    h, w = img.shape[:2]
    regions = {
        "Top-left": img[0:h//2, 0:w//2],
        "Top-right": img[0:h//2, w//2:w],
        "Bottom-left": img[h//2:h, 0:w//2],
        "Bottom-right": img[h//2:h, w//2:w],
        "Center": img[h//4:3*h//4, w//4:3*w//4]
    }
    
    print(f"\nTesting different regions:")
    for name, crop in regions.items():
        clothing, _ = desc.describe(crop)
        print(f"  {name:15s}: {clothing}")
    
    print(f"\n{'='*70}")
    print("✓ Test complete!")
    print(f"{'='*70}")
else:
    print("\n✗ test_image.jpg not found")
    print("  Place a test image in the current directory")
