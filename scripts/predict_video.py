import tensorflow as tf
import numpy as np
import cv2
import os

# 模型路径（修改成你自己的路径）
MODEL_DIR = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/F_ExportModel/saved_model"
VIDEO_PATH = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/Jasmine_milk_tea_2nd_pour.mp4"
OUTPUT_PATH = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/Jasmine_milk_tea_2nd_pour_RR.mp4"

# 类别标签字典
CATEGORY_INDEX = {
    1: {'id': 1, 'name': 'full'},
    2: {'id': 2, 'name': 'not_full'},
    3: {'id': 3, 'name': 'foam_ready'}
}

print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_DIR)
print("Model loaded.")

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

CONF_THRESH = 0.3
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # 只取 top-1 且分数大于阈值的预测
    if scores[0] > CONF_THRESH:
        y1, x1, y2, x2 = boxes[0]
        (left, top, right, bottom) = (x1 * width, y1 * height, x2 * width, y2 * height)
        class_id = classes[0]
        label = CATEGORY_INDEX.get(class_id, {'name': 'unknown'})['name']
        score = scores[0]
        text = f"{label}: {score:.2f}"
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        cv2.putText(frame, text, (int(left), int(top) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"\n✅ Detection finished. Output saved to: {OUTPUT_PATH}")