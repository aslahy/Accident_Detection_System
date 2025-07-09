import os
import cv2

# Root of your dataset
dataset_root = "data"
output_dir = "accident_crops"
os.makedirs(output_dir, exist_ok=True)

# Loop over all dataset splits
splits = ['train', 'valid', 'test']

for split in splits:
    image_dir = os.path.join(dataset_root, split, "images")
    label_dir = os.path.join(dataset_root, split, "labels")

    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip malformed lines

                class_id, x_c, y_c, bw, bh = map(float, parts)

                # Convert YOLO format to pixel coordinates
                x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
                x1 = int(x_c - bw / 2)
                y1 = int(y_c - bh / 2)
                x2 = int(x_c + bw / 2)
                y2 = int(y_c + bh / 2)

                # Clip to bounds
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w - 1), min(y2, h - 1)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_name = f"{split}_{os.path.splitext(image_name)[0]}_{i}.jpg"
                cv2.imwrite(os.path.join(output_dir, crop_name), crop)
