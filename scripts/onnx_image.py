import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
session = ort.InferenceSession(
    "./model.onnx"
)

# Print input and output names
print("Inputs:", [i.name for i in session.get_inputs()])
print("Outputs:", [o.name for o in session.get_outputs()])

# Load test image
image = cv2.imread("")  # Replace with any test image
input_tensor = cv2.resize(image, (640, 640))  # Resize to match model input size
input_tensor = input_tensor.astype(np.uint8)
input_tensor = np.expand_dims(input_tensor, axis=0)

# Run inference
outputs = session.run(None, {"input_tensor": input_tensor})

# Extract detection results
boxes = outputs[1][0]           # detection_boxes
classes = outputs[2][0].astype(np.int32)
scores = outputs[4][0]          # detection_scores

# Class label map
category_index = {
    1: "full",
    2: "not_full",
    3: "foam_ready"
}

# Draw bounding boxes and labels
for i in range(len(scores)):
    if scores[i] > 0.5:
        box = boxes[i]
        y1, x1, y2, x2 = box
        h, w, _ = image.shape
        left, top, right, bottom = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        label = category_index.get(classes[i], str(classes[i]))
        cv2.putText(image, f"{label}: {scores[i]:.2f}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Show result
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()