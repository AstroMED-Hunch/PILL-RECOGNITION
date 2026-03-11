# PILL-RECOGNITION-SOLUTION
disclaimer: This is a presentable repo - the pill recognition is NOT separate from our backend - [check out the backend repo for the full codebase](https://github.com/AstroMED-Hunch/Backend) - although we do still include the model & basic test code in this repo)

# Pill Detection Model ( .ONNX )

Trained on the [NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC5973812/) dataset using YOLO11s. Exported to ONNX for integration into C++ backend via ONNX Runtime.

-----

## Model Overview

|Property       |Value                                   |
|---------------|----------------------------------------|
|Architecture   |YOLO11s (Leaky ReLU variant)            |
|Framework      |Ultralytics 8.4.x                       |
|Task           |Object Detection                        |
|Classes        |1 (`pill`) - class_id: 0                |
|Input shape    |`[1, 3, 480, 832]` NCHW float32         |
|Output shape   |`[1, 300, 6]`                           |
|Output format  |`[x1, y1, x2, y2, confidence, class_id]`|
|Input range    |0.0 – 1.0 (divide raw pixels by 255)    |
|Training epochs|120                                     |
|Image size     |480 × 832                               |

-----

## Output Format

Each inference call returns a tensor of shape `[1, 300, 6]`. Each of the 300 rows represents one candidate detection:

```
[x1, y1, x2, y2, confidence, class_id]
```

- `x1, y1, x2, y2` - bounding box corners in model input coordinates (scale back to original image size)
- `confidence` - detection confidence score between 0 and 1
- `class_id` - always `0` (pill)

Rows with confidence below your chosen threshold should be discarded. A threshold of `0.40` is a reasonable starting point.

-----

## Training

covers pill images across varied lighting, backgrounds, and orientations with bounding box annotations in YOLO format.

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
    nms=True,
    simplify=True,
    dynamic=False,
    opset=17,
    half=False,
)
```

The `nms=True` flag bakes Non-Maximum Suppression into the graph so the output is already post-processed. No additional NMS step is needed at inference time.

-----
-----

## Test Results

Tested on pill images across capsules, tablets, and coated pills.

(video link will be placed here).

-----

## C++ Integration for live individual pill recognition

The model integrates into C++ via [ONNX Runtime](https://onnxruntime.ai/) and OpenCV. The preprocessing pipeline is straightforward:

TBD 

-----

## Files

```
astromed.onnx  # ONNX model 
```

-----

## Dependencies

|Library     |Version|
|------------|-------|
|Ultralytics |8.4.x  |
|ONNX        |1.20.x |
|ONNX Runtime|1.18+  |
|OpenCV      |4.x    |
|Python      |3.10+  |

-----

## License

YOLO11 base architecture is released under the [Ultralytics AGPL-3.0 license](https://ultralytics.com/license).
