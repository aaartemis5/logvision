"""
Capture person crops from surveillance video to see what SigLIP is analyzing.
This runs independently without modifying the main pipeline.
"""
import cv2
import os
from ultralytics import YOLO
from visionlog import config
from visionlog.enrichment.siglip_describer import get_siglip_describer

print("="*80)
print("CAPTURING PERSON CROPS FOR SIGLIP ANALYSIS")
print("="*80)

# Create debug directory
debug_dir = "debug_crops"
os.makedirs(debug_dir, exist_ok=True)
print(f"\nSaving crops to: {debug_dir}/")

# Load YOLO
print("\nLoading YOLO model...")
yolo = YOLO(config.YOLO_MODEL)

# Get first camera
cam_id = list(config.CAMERAS.keys())[0]
cam_cfg = config.CAMERAS[cam_id]
video_path = cam_cfg['path']

print(f"Camera: {cam_cfg['name']}")
print(f"Video: {video_path}")

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"\n✗ Could not open video")
    exit(1)

# Skip to middle of video for better variety
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

# Read frame
ret, frame = cap.read()
if not ret:
    print("\n✗ Could not read frame")
    exit(1)

print(f"\nFrame size: {frame.shape}")

# Detect people
print("\nDetecting people...")
results = yolo.track(frame, persist=True, classes=[0], conf=config.YOLO_CONF, verbose=False)

if len(results[0].boxes) == 0:
    print("\n✗ No people detected")
    exit(1)

print(f"✓ Found {len(results[0].boxes)} people")

# Get SigLIP
desc = get_siglip_describer()

# Process each person
print("\n" + "="*80)
print("ANALYZING EACH PERSON")
print("="*80)

h, w = frame.shape[:2]
annotated_frame = frame.copy()

for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Full person crop
    full_crop = frame[y1:y2, x1:x2]
    
    # Upper 70% crop (what SigLIP actually sees)
    ub = min(h, y1 + int((y2-y1)*0.70))
    siglip_crop = frame[y1:ub, x1:x2]
    
    if siglip_crop.size == 0:
        continue
    
    # Detect clothing
    clothing, emb = desc.describe(siglip_crop)
    
    # Save crops
    full_path = f"{debug_dir}/person_{i+1}_full.jpg"
    crop_path = f"{debug_dir}/person_{i+1}_upper70.jpg"
    
    cv2.imwrite(full_path, full_crop)
    cv2.imwrite(crop_path, siglip_crop)
    
    print(f"\nPerson {i+1}:")
    print(f"  Bbox: ({x1},{y1}) → ({x2},{y2})")
    print(f"  Full crop: {full_crop.shape} → {full_path}")
    print(f"  Upper 70%: {siglip_crop.shape} → {crop_path}")
    print(f"  DETECTED: {clothing}")
    
    # Draw on frame
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(annotated_frame, (x1, y1), (x2, ub), (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"#{i+1}: {clothing}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save annotated frame
annotated_path = f"{debug_dir}/annotated_frame.jpg"
cv2.imwrite(annotated_path, annotated_frame)

print("\n" + "="*80)
print("SAVED FILES:")
print("="*80)
print(f"\n✓ {annotated_path}")
print(f"   → Full frame with all detections")

for i in range(len(results[0].boxes)):
    print(f"\n✓ Person {i+1}:")
    print(f"   → {debug_dir}/person_{i+1}_full.jpg (full person)")
    print(f"   → {debug_dir}/person_{i+1}_upper70.jpg (what SigLIP sees)")

print("\n" + "="*80)
print("NOW CHECK:")
print("="*80)
print("\n1. Open the person_X_upper70.jpg files")
print("2. Compare with the detected clothing")
print("3. See if the crops make sense")
print("4. Check if colors are visible in the crops")
print("\n" + "="*80)

cap.release()
