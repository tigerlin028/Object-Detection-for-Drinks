# Object Detection for Drinks

This repository contains the training pipeline, scripts, and configuration files used to train a custom object detection model for drink volume classification using TensorFlow 2.

## Model Architecture

- **Base model**: `ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8`
- **Input size**: 640×640
- **Framework**: TensorFlow 2.x Object Detection API

## Project Structure

```
├── core/          # Training logic
├── configs/       # Model config
├── data/          # TFRecords, label maps
├── scripts/       # Inference & utils
```

## Usage

### Train

```bash
python core/model_main_tf2.py --model_dir=... --pipeline_config_path=... --alsologtostderr
```

### Export

```bash
python scripts/exporter_main_v2.py --input_type image_tensor --pipeline_config_path=... --trained_checkpoint_dir=... --output_directory=...
```

### Predict

```bash
python scripts/predict.py
python scripts/predict_video.py
```

---

Built by tigerlin028