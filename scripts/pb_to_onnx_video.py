import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
session = ort.InferenceSession(
    "C:/Users/Josh Huang/Documents/TensorFlow/models/research/data/models/F_ExportModel/model.onnx"
)

# Class label map
category_index = {
    1: "full",
    2: "not_full",
    3: "foam_ready"
}

# Non-Maximum Suppression function
def nms(boxes, scores, iou_threshold=0.5):
    indices = []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        indices.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou < iou_threshold]

    return indices

# Load input video
video_path = "Brown_sugar_milk.mp4"
cap = cv2.VideoCapture(video_path)

# Save output video
save_output = True
output_path = "Brown_sugar_milk_R.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) if save_output else None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and prepare input
    input_tensor = cv2.resize(frame, (640, 640))  # Adjust to model input size
    input_tensor = input_tensor.astype(np.uint8)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Run inference
    outputs = session.run(None, {"input_tensor": input_tensor})

    boxes = outputs[1][0]
    classes = outputs[2][0].astype(np.int32)
    scores = outputs[4][0]

    valid_boxes = []
    valid_scores = []
    valid_classes = []

    for i in range(len(scores)):
        if scores[i] > 0.5:
            valid_boxes.append(boxes[i])
            valid_scores.append(scores[i])
            valid_classes.append(classes[i])

    keep = nms(valid_boxes, valid_scores, iou_threshold=0.5)

    for i in keep:
        box = valid_boxes[i]
        cls = valid_classes[i]
        score = valid_scores[i]

        y1, x1, y2, x2 = box
        h, w, _ = frame.shape
        left, top, right, bottom = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        label = category_index.get(cls, str(cls))
        cv2.putText(frame, f"{label}: {score:.2f}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Result", frame)
    if save_output:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()