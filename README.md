# 🥤 Object Detection for Drink Fill Level

This project demonstrates a full TensorFlow 2.x pipeline for training and deploying a custom object detection model to detect drink fill levels using the SSD MobileNet V2 architecture.

---

## 🚀 Workflow Overview

### 1. Environment Setup
- Create a Conda environment (e.g. `tfod_py39`) with TensorFlow 2.10.1.
- Install dependencies for the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
- Download the base model:
  ```
  ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8
  ```

---

### 2. Data Preparation
- Use `CVAT` to:
  - Annotate bounding boxes on each frame of your MP4 video.
  - Export the dataset in **Pascal VOC** format.
- Extract frames from the video using CVAT or ffmpeg, save them as `.png`.
- Organize the directory like this:
  ```
  JPEGImages/
  Annotations/
  ImageSets/
  ```

---

### 3. TFRecord Generation
- Use the provided script:
  ```bash
  python scripts/generate_tfrecord_png.py
  ```
- This will create `train.record` and `test.record` files from VOC annotations.

---

### 4. Model Training
- Modify `configs/pipeline.config` to:
  - Set your `num_classes`
  - Point to the correct `label_map_path` and TFRecord paths
  - Use the correct fine-tune checkpoint

- Start training:
  ```bash
  python core/model_main_tf2.py \
    --model_dir=data/models/custom_drink_dataModel \
    --pipeline_config_path=configs/pipeline.config \
    --alsologtostderr
  ```

---

### 5. Export Trained Model
- Use the following script to export your model:
  ```bash
  python core/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path=configs/pipeline.config \
    --trained_checkpoint_dir=custom_drink_dataModel \
    --output_directory=F_ExportModel
  ```

---

### 6. Run Inference on Images or Videos (TensorFlow SavedModel)

- Predict bounding boxes on a single image using SavedModel:
  ```bash
  python scripts/pb_image.py
  ```

- Overlay predictions on an entire video using SavedModel:
  ```bash
  python scripts/pb_video.py
  ```

---

### 7. Convert TensorFlow Model to ONNX

After exporting your model in TensorFlow `saved_model` format, convert it to ONNX with:

```bash
python -m tf2onnx.convert ^
  --saved-model "F_ExportModel/saved_model" ^
  --output "F_ExportModel/model.onnx" ^
  --opset 13 ^
  --tag serve ^
  --signature_def serving_default
```

Make sure to install the required packages:
```bash
pip install tf2onnx onnx
```

This produces an optimized `model.onnx` file ready for efficient inference using ONNX Runtime.

---

### 8. Run Inference on Images or Videos (ONNX)

- Predict bounding boxes on a **single image** using ONNX:
  ```bash
  python scripts/onnx_image.py
  ```

- Predict bounding boxes on a **video** using ONNX with NMS:
  ```bash
  python scripts/onnx_video.py
  ```

---

## 📂 Project Structure

```
├── core/                  # Training scripts (model_main, model_lib, exporter)
├── configs/               # Model pipeline configuration
├── data/                  # TFRecords and label map
├── scripts/               # Inference and conversion utilities
│   ├── generate_tfrecord_png.py
│   ├── pb_image.py
│   ├── pb_video.py
│   ├── onnx_image.py
│   └── onnx_video.py
├── .gitignore
├── README.md
```

---

## 🧠 Key Tools & Concepts

- **TensorFlow 2 Object Detection API**
- **ONNX Runtime for deployment**
- **SSD MobileNet V2 FPN 640x640**
- **CVAT for annotation**
- **TFRecord for input pipelines**
- **Real-time bounding box rendering using OpenCV**

---

## ✍️ Author

Developed by Xiaotian Lin [@tigerlin028](https://github.com/tigerlin028)