import cv2
import os
import time
import psutil
import pandas as pd
from modules.feature_extractor import SIFT, ORB, BRISK  # replace with your file/module name

# === CONFIG ===
image_folder = "/media/usb/masfiles/test_images_mh/images/data"  # path to folder of images
extractor_type = "SIFT"  # choose from "SIFT", "ORB", "BRISK"
resize_width = 320    # set to None to skip resizing
resize_height = 240   # set to None to skip resizing
log_csv = "feature_benchmark_results_sift.csv"  # CSV file to save results
max_images = 75

# === Initialize extractor ===
if extractor_type == "SIFT":
    extractor = SIFT(n_features=500)
elif extractor_type == "ORB":
    extractor = ORB(n_features=500)
elif extractor_type == "BRISK":
    extractor = BRISK()
else:
    raise ValueError("Unknown extractor type")

# === Load and sort images ===
images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))])

# Limit number of images if specified
if max_images is not None:
    images = images[:max_images]

if len(images) < 2:
    print("Need at least two images in the folder")
    exit()

# === Benchmark ===
results = []
prev_kp, prev_des = None, None

for i, img_path in enumerate(images):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        continue

    # Resize if needed
    if resize_width is not None and resize_height is not None:
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

    # CPU & memory before processing
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Keypoint detection + descriptor computation
    start_time = time.time()
    kp, des = extractor.detect_and_compute(img)
    detection_time = time.time() - start_time
    num_kp = len(kp)

    # Feature matching
    if prev_des is not None:
        start_time = time.time()
        matches = extractor.match_features(prev_des, des)
        matching_time = time.time() - start_time
        num_matches = len(matches)
    else:
        matching_time = 0
        num_matches = 0

    # CPU & memory after processing
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Save results to list
    results.append({
        "image": os.path.basename(img_path),
        "keypoints": num_kp,
        "good_matches": num_matches,
        "detection_time": round(detection_time, 3),
        "matching_time": round(matching_time, 3),
        "cpu_percent": cpu_after,
        "memory_mb": round(mem_after, 2)
    })

    prev_kp, prev_des = kp, des

# === Convert to pandas DataFrame and save ===
df = pd.DataFrame(results)
df.to_csv(log_csv, index=False)
print(f"\nAll results saved to {log_csv}")

# === Print summary ===
print(f"\nBenchmark results for {extractor_type} on {len(images)} images:")
print(df)

# Compute averages
avg_kp = df["keypoints"].mean()
avg_matches = df["good_matches"].iloc[1:].mean()  # skip first image
avg_det_time = df["detection_time"].mean()
avg_match_time = df["matching_time"].iloc[1:].mean()
avg_cpu = df["cpu_percent"].mean()
avg_mem = df["memory_mb"].mean()

print(f"\nAverages -> Keypoints: {avg_kp:.1f}, Matches: {avg_matches:.1f}, "
      f"Detection time: {avg_det_time:.3f}s, Matching time: {avg_match_time:.3f}s, "
      f"CPU: {avg_cpu:.1f}%, Memory: {avg_mem:.2f}MB")