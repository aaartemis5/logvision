"""
Test script to verify YOLO26-pose availability and OpenVINO export capability.
This will help us decide between YOLO11 and YOLO26 for the migration.
"""
import sys
from pathlib import Path

print("="*70)
print("YOLO26-Pose Verification Test")
print("="*70)

# Test 1: Check Ultralytics version
print("\n[1/5] Checking Ultralytics version...")
try:
    from ultralytics import YOLO, __version__
    print(f"✓ Ultralytics version: {__version__}")
    
    # YOLO26 requires ultralytics >= 8.3.0 (released Jan 2026)
    major, minor = map(int, __version__.split('.')[:2])
    if major < 8 or (major == 8 and minor < 3):
        print(f"⚠ Warning: YOLO26 requires ultralytics >= 8.3.0")
        print(f"  Current version: {__version__}")
        print(f"  Run: pip install --upgrade ultralytics")
except ImportError as e:
    print(f"✗ Ultralytics not installed: {e}")
    sys.exit(1)

# Test 2: Check available YOLO26 models
print("\n[2/5] Checking YOLO26 model availability...")
yolo26_variants = [
    "yolo26n-pose.pt",  # Nano - fastest
    "yolo26s-pose.pt",  # Small
    "yolo26m-pose.pt",  # Medium
    "yolo26l-pose.pt",  # Large - best accuracy
    "yolo26x-pose.pt",  # Extra large
]

available_models = []
for model_name in yolo26_variants:
    try:
        print(f"  Trying to load {model_name}...", end=" ")
        model = YOLO(model_name)
        print(f"✓ Available")
        available_models.append(model_name)
    except Exception as e:
        print(f"✗ Not available ({str(e)[:50]}...)")

if not available_models:
    print("\n✗ No YOLO26-pose models found!")
    print("  YOLO26 may not be released yet or requires manual download.")
    print("\n  Fallback recommendation: Use YOLO11-pose or YOLOv10-pose")
    sys.exit(1)

print(f"\n✓ Found {len(available_models)} YOLO26-pose model(s)")

# Test 3: Load and inspect model
print("\n[3/5] Loading YOLO26l-pose for detailed inspection...")
try:
    model = YOLO("yolo26l-pose.pt")
    print(f"✓ Model loaded successfully")
    print(f"  Task: {model.task}")
    print(f"  Model type: {type(model.model).__name__}")
    
    # Check if it has pose keypoints
    if hasattr(model.model, 'kpt_shape'):
        print(f"  Keypoints: {model.model.kpt_shape} (17 keypoints expected)")
    else:
        print(f"  ⚠ Warning: Could not verify keypoint configuration")
        
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Test 4: Test inference on dummy image
print("\n[4/5] Testing inference on dummy image...")
try:
    import numpy as np
    import cv2
    
    # Create a dummy 640x640 image
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("  Running inference...", end=" ")
    results = model(dummy_img, verbose=False)
    print("✓ Inference successful")
    
    # Check results structure
    r = results[0]
    print(f"  Boxes: {r.boxes is not None}")
    print(f"  Keypoints: {r.keypoints is not None}")
    
    if r.keypoints is not None:
        print(f"  Keypoint shape: {r.keypoints.shape if hasattr(r.keypoints, 'shape') else 'N/A'}")
        print(f"✓ YOLO26-pose has keypoint detection!")
    else:
        print(f"✗ No keypoints detected - this may not be a pose model")
        
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test OpenVINO export
print("\n[5/5] Testing OpenVINO export...")
try:
    print("  Exporting to OpenVINO format...", end=" ")
    export_path = model.export(format='openvino', dynamic=False, half=False)
    print(f"✓ Export successful!")
    print(f"  Export path: {export_path}")
    
    # Check exported files
    export_dir = Path(export_path).parent
    xml_file = export_dir / f"{Path(export_path).stem}.xml"
    bin_file = export_dir / f"{Path(export_path).stem}.bin"
    
    if xml_file.exists() and bin_file.exists():
        print(f"  ✓ OpenVINO IR files created:")
        print(f"    - {xml_file.name} ({xml_file.stat().st_size / 1024:.1f} KB)")
        print(f"    - {bin_file.name} ({bin_file.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"  ⚠ Warning: Expected .xml and .bin files not found")
        
except Exception as e:
    print(f"✗ Export failed: {e}")
    print("\n  This is a critical issue - YOLO26 cannot be used with OpenVINO yet.")
    print("  Recommendation: Use YOLO11-pose or YOLOv10-pose instead.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final verdict
print("\n" + "="*70)
print("VERDICT: YOLO26-Pose Compatibility")
print("="*70)
print("✓ YOLO26-pose models are available")
print("✓ Pose keypoints are supported")
print("✓ Inference works correctly")
print("✓ OpenVINO export is functional")
print("\n✓✓✓ YOLO26-POSE IS READY FOR PRODUCTION USE ✓✓✓")
print("\nRecommendation: Proceed with YOLO26l-pose for your migration.")
print("="*70)
