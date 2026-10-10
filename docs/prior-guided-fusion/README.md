# Prior Guided Fusion for Crack Detection

This project explores incorporating handcrafted texture priors into YOLO-based
object detection architectures for crack localization.

## Texture Anomaly Map

The texture-prior pipeline combines directional Gabor responses, multiscale
smoothing, high-frequency residual extraction, edge suppression, and normalization.

![Texture Anomaly Map](images/texture-anomaly-map.png)

## Prior Gated Fusion

The PGF module combines a projected prior map with backbone features through
a learned, feature-dependent gate.

![Prior Gated Fusion](images/pgf-module.png)

## Model Variants

### YOLOv8 with Scribble-Derived Prior
![YOLOv8 Scribble Prior](images/yolov8-scribble-prior.png)

### YOLOv8 with Texture Anomaly Prior
![YOLOv8 Texture Prior](images/yolov8-texture-prior.png)

### YOLOv12 with Texture Anomaly Prior
![YOLOv12 Texture Prior](images/yolov12-texture-prior.png)
