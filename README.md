# Weighted-Boxes-Fusion-implementation

## About
This is repository for implementing bounding box ensemble method (weighted-boxes-fusion) with multiple detection models (YOLOv4 and YOLOv5)
Also, repo was created for the purpose of participating in the hackathon competition.

## Configuration

## Result
||Precision|Recall|$mAP_{0.5}|$mAP_{.5 - .95}|
|---|---|---|---|---|
|YOLOv4|0.5261|0.8096|0.696|0.5838|
|YOLOv5|0.6018|0.7129|0.5344|0.6673|
|Ensemble|0.8158|0.8988|0.9192|0.8184|

## Reference
Weighted-boxes-fusion: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
YOLOv4: https://github.com/WongKinYiu/PyTorch_YOLOv4
YOLOv5: https://github.com/ultralytics/yolov5
