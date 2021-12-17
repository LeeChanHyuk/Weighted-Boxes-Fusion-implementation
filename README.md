# Weighted-Boxes-Fusion-implementation

## About
This is repository for implementing bounding box ensemble method (weighted-boxes-fusion) with multiple detection models (YOLOv4 and YOLOv5)

Also, repo was created for the purpose of participating in the hackathon competition.

## directory structure
Weighted-Boxes-Fusion-implementation
  - additional_utils
  - concat (main folder)
    - yolov5 (yolov5 model)
      - data (data.yaml folder)
      - models
      - yolov4 (yolo v4 model)
        - cfg
        - models 
        - v4_data
        - v4_utils
        - weights
      - WBF
      - utils
  - dataset (dataset folder. You must fill in this directory with your dataset)
    - annotations
    - images
    - labels

## Make your dataset

you can organize your dataset folder with this cite (https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## Training 
### YOLO V4 Training

```python
  cd ROOT/concat/yolov5/yolov4
  python train.py --weights yolov4.weight --data v4_data/custon.yaml
```

### YOLO V5 Training

```python
  cd ROOT/concat/yolov5
  python train.py --weights yolov5s.pt --data data/custon.yaml
```

## Result check
if you want to check my result with hackathon dataset, please follow the below direction.

1. please make the hackathon dataset in dataset folder.
2. Download the weights
  - yolov4 (https://drive.google.com/file/d/1XBls41JzDdbUC5SvPbUoahCXZ-KyLR-w/view?usp=sharing)
  - yolov5 (https://drive.google.com/file/d/1QhExYLCD8Wc4sf7XvWcIdy1zIK2B2j3k/view?usp=sharing)

and put the yolov4 weight in concat/yolov5/yolov4/weights
put the yolov5 weight in concat/yolov5

3. implement the test code in yolov5 folder
   ```python
    cd ROOT/concat/yolov5
    python test.py --data your_yamlpath.yaml --yolov4_weight v4_best.pt --yolov5_weight v5_best.pt yolov4_cfg yolov4/cfg/yolov4-pacsp-x.cfg
   ```
   

## Result
||Precision|Recall|$mAP~0.5~|$mAP~.5 - .95~|
|---|---|---|---|---|
|YOLOv4|0.5261|0.8096|0.696|0.5838|
|YOLOv5|0.6018|0.7129|0.5344|0.6673|
|Ensemble|0.8158|0.8988|0.9192|0.8184|

## Reference
Weighted-boxes-fusion: https://github.com/ZFTurbo/Weighted-Boxes-Fusion

YOLOv4: https://github.com/WongKinYiu/PyTorch_YOLOv4

YOLOv5: https://github.com/ultralytics/yolov5
