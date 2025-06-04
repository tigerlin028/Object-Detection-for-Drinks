import tensorflow as tf
import numpy as np
import cv2
import os

# 模型路径（改成你自己的）
MODEL_DIR = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/CExportModel/saved_model"
VIDEO_PATH = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/Peach_paradise_Lychee_chia_etc.mp4"
OUTPUT_PATH = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/Peach_paradise_Lychee_chia_etc_R.mp4"

# 类别标签字典（修改成你自己的）
CATEGORY_INDEX = {
    1: {'id': 1, 'name': 'full'},
    2: {'id': 2, 'name': 'notfull'}
}

# 加载模型
print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_DIR)
print("Model loaded.")

# 视频读取初始化
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 输出视频编码器设置
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

    for i in range(min(100, boxes.shape[0])):
        if scores[i] > CONF_THRESH:
            y1, x1, y2, x2 = boxes[i]
            (left, top, right, bottom) = (x1 * width, y1 * height, x2 * width, y2 * height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = CATEGORY_INDEX.get(classes[i], {'name': 'unknown'})['name']
            text = f"{label}: {scores[i]:.2f}"
            cv2.putText(frame, text, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"Video saved to {OUTPUT_PATH}")