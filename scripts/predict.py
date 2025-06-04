import tensorflow as tf
import numpy as np
import cv2
import os

# 模型路径（改成你实际的路径）
MODEL_DIR = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/CExportModel/saved_model"
# 图片路径（换成你自己的测试图）
IMAGE_PATH = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/cb2.png"
# 结果保存路径
OUTPUT_PATH = "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/cbresult2.png"
# 标签（你只有一个类，可直接写）
CATEGORY_INDEX = {
    1: {'id': 1, 'name': 'full'},
    2: {'id': 2, 'name': 'notfull'}
}

# 加载模型
print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_DIR)
print("Model loaded.")

# 加载图片
image_np = cv2.imread(IMAGE_PATH)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

# 运行推理
detections = detect_fn(input_tensor)

# 解析结果
boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)

# 阈值
CONF_THRESH = 0.1

# 可视化结果
height, width, _ = image_np.shape
for i in range(min(100, boxes.shape[0])):
    if scores[i] > CONF_THRESH:
        y1, x1, y2, x2 = boxes[i]
        (left, top, right, bottom) = (x1 * width, y1 * height, x2 * width, y2 * height)
        cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        label = CATEGORY_INDEX[classes[i]]['name']
        score = scores[i]
        text = f"{label}: {score:.2f}"
        cv2.putText(image_np, text, (int(left), int(top) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存结果
cv2.imwrite(OUTPUT_PATH, image_np)
print(f"Detection completed. Result saved to {OUTPUT_PATH}")
