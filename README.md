# PILL-RECOGNITION-SOLUTION
disclaimer: This is a presentable repo - the pill recognition is NOT separate from our backend - [check out the backend repo for the full codebase](https://github.com/AstroMED-Hunch/Backend) - although we do still include the model & basic test code in this repo)
# Pill Detection Model ( .ONNX )
Trained on the [NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC5973812/) dataset using YOLO11s. Exported to ONNX for integration into C++ backend via OpenCV dnn.
-----
## Model Overview
|Property       |Value                                   |
|---------------|----------------------------------------|
|Architecture   |YOLO11s (Leaky ReLU variant)            |
|Framework      |Ultralytics 8.4.x                       |
|Task           |Object Detection                        |
|Classes        |1 (`pill`) - class_id: 0                |
|Input shape    |`[1, 3, 480, 832]` NCHW float32         |
|Output shape   |`[1, 5, 8190]`                          |
|Output format  |`[cx, cy, w, h, confidence]` per anchor |
|Input range    |0.0 – 1.0 (divide raw pixels by 255)    |
|Training epochs|120                                     |
|Image size     |480 × 832                               |
-----
## Output Format
Each inference call returns a tensor of shape `[1, 5, 8190]`. Transpose to `[8190, 5]` so each row is one anchor:
```
[cx, cy, w, h, confidence]
```
- `cx, cy` - bounding box center in model input coordinates
- `w, h` - bounding box dimensions in model input coordinates
- `confidence` - detection confidence score between 0 and 1

Scale box coordinates back to original image size and apply NMS manually. A confidence threshold of `0.40` and IoU threshold of `0.45` are our starting points.
-----
## Training
Covers pill images across varied lighting, backgrounds, and orientations with bounding box annotations in YOLO format.

**Training config:**
```yaml
model:    astromed.yaml   # YOLO11s with Leaky ReLU activations
epochs:   120
patience: 100
imgsz:    [480, 832]
batch:    32
optimizer: auto
augment:   randaugment
mosaic:    1.0
fliplr:    0.5
hsv_s:    0.7
hsv_v:    0.4
```
**Export:**
```python
from ultralytics import YOLO
model = YOLO("astromed.pt")
model.export(
    format="onnx",
    imgsz=[480, 832],
    nms=False,
    simplify=True,
    dynamic=False,
    opset=17,
    half=False,
)
```
NMS is left off so the C++ backend can apply it directly via `cv::dnn::NMSBoxes` after filtering by confidence threshold.
-----
## Test Results
Tested on pill images across capsules, tablets, and coated pills.

[(videos here).](https://sites.google.com/pcti.mobi/astromed/post-cdr?authuser=0)
-----
## C++ Integration for live individual pill recognition
The model loads via `cv::dnn::readNet` and runs as a module inside the backend. Each frame is passed into `PillDetector::run()` which handles the full pipeline: blob creation, forward pass, output parsing, NMS, and per-detection ORB feature matching against a directory of registered reference images.

**Preprocessing:**
```cpp
cv::Mat blob = cv::dnn::blobFromImage(
    frame, 1.0 / 255.0, cv::Size(832, 480), cv::Scalar(), true, false, CV_32F
);
net.setInput(blob);
cv::Mat raw = net.forward();
```

**Output parsing:**

The raw output is shaped `[1, 5, 8190]`. The backend transposes it to `[8190, 5]` so each row is one anchor `[cx, cy, w, h, confidence]`, filters by confidence, converts center-format boxes to `cv::Rect`, then runs `cv::dnn::NMSBoxes` to remove overlapping detections.

**Individual pill identification:**

Once pill regions are cropped from the frame, each crop is run through ORB feature extraction and matched against every reference image in the configured `pill_images_dir`. The reference image filename is used as the pill name. The detection with the highest match score wins. If no reference images are loaded the pill is labeled `misc`.

**Registration** works the same way as crew member face registration in the rest of the backend — drop a reference image named after the pill into the directory and it is picked up automatically on next initialization:
```
extern/pills/aspirin.jpg
extern/pills/ibuprofen.jpg
extern/pills/paracetamol.jpg
```

**Config keys** (set in layout config):
```
pill_detector_model   path to astromed.onnx
pill_images_dir       path to reference image directory
```
-----
## Files
```
astromed.onnx  # ONNX model
test.py        # basic python test script
```
-----
## Dependencies
|Library     |Version|
|------------|-------|
|Ultralytics |8.4.x  |
|ONNX        |1.20.x |
|OpenCV      |4.x    |
|Python      |3.10+  |
-----
## License
YOLO11 base architecture is released under the [Ultralytics AGPL-3.0 license](https://ultralytics.com/license).
