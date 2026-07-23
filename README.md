# SimpleObjectDetectionBY_yolo

This project trains a **YOLOv8 classification model** (`yolov8n-cls.pt`) on custom images captured from personal devices.

> Note: despite the repository name, the current notebook workflow is image **classification** (not bounding-box object detection).

## Repository Contents

- `assignment_8_yolo_task1.ipynb` – end-to-end Colab workflow:
  - install dependencies
  - mount Google Drive
  - prepare train/validation folders
  - split images
  - train YOLOv8 classifier
  - run validation and prediction
- `README.md` – project documentation

## Classes Used

The notebook trains on the following classes:

- `sky`
- `road`
- `flower`
- `nature`

## Requirements

- Python 3.8+
- [Ultralytics](https://docs.ultralytics.com/)
- PyTorch (with CUDA recommended for faster training)
- Google Colab + Google Drive (as used in the notebook)

Install the main dependency:

```bash
pip install ultralytics
```

## Dataset Layout

The notebook expects source images in:

```text
/content/drive/MyDrive/assignment /Pictures
```

It creates this training structure automatically:

```text
/content/dataset_cls/
  train/
    sky/
    road/
    flower/
    nature/
  val/
    sky/
    road/
    flower/
    nature/
```

## Training

The model is trained with:

- model: `yolov8n-cls.pt`
- epochs: `30`
- image size: `224`
- batch size: `16`

Equivalent training call used in the notebook:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")
model.train(data="/content/dataset_cls", epochs=30, imgsz=224, batch=16)
```

## Validation and Prediction

After training:

1. Validate model:
   ```python
   metrics = model.val()
   ```
2. Predict on a test image:
   ```python
   results = model.predict(source="path/to/image.jpg", save=True)
   ```
3. Read top class and confidence:
   ```python
   r = results[0]
   print(r.names[r.probs.top1], r.probs.top1conf)
   ```

Prediction output images are saved under:

```text
/content/runs/classify/predict/
```

## How to Run

1. Open `assignment_8_yolo_task1.ipynb` in Google Colab.
2. Update `IMAGE_DIR` to your dataset location in Drive.
3. Run notebook cells from top to bottom.
4. Review metrics and prediction outputs.
