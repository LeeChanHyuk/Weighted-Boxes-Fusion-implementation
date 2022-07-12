# Weighted-Boxes-Fusion-implementation

## Weighted-Boxes-Fusion ?
  **WBF (Weighted-Boxes-Fusion) is the method combining predictions of object detection models.**
  
  *Original paper: Weighted boxes fusion: Ensembling boxes from different object detection models (Image and Vision Computing, 2021)*

## About this repo
This is repository for implementing bounding box ensemble method (weighted-boxes-fusion) with multiple detection models (YOLOv4 and YOLOv5).
The overall process for using weighted-boxes-fusion method is described in below section.
If you have an issue of using my code, please make issue on my repo.
**If my repo could help you, your one star can help me a lot. Thanks**

And note that this repo was created for the purpose of participating in the hackathon competition.
(Competition info: Illegal object detection part from Busan metropolitan City artificial intelligence model competition)

## Install

```python
  pip install -r requirements.txt
```

## (Optional) Import virtual conda virtual environment (Recommended in Linux)

```python
  cd ROOT
  conda env create -f test_env.yaml
```

## directory structure
Weighted-Boxes-Fusion-implementation
  - additional_utils
  - model
    - yolov5 (Example model 1)
      - data (data.yaml folder)
      - models
      - WBF
      - utils
    - yolov4 (Example model 2)
      - cfg
      - models 
      - v4_data
      - v4_utils
      - weights
  - dataset (dataset folder. You must fill in this directory with your dataset)
    - annotations
    - images
    - labels
    - dataset_yaml
  - configuration.yaml (If you want to use your models, please change the configuration.yaml with below recommendation.)

## Make your dataset

you can organize your dataset folder with this cite (https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## Training 
### YOLO V4 Training

```python
  cd ROOT/model/yolov4 # ROOT is the root directory of your project
  python train.py --weights yolov4.weight --data v4_data/custon.yaml
```

### YOLO V5 Training

```python
  cd ROOT/model/yolov5
  python train.py --weights yolov5s.pt --data data/custon.yaml
```

## Test
If you want to test wbf result with your models, please put the weight file of your model in the wbf folders.
Also, change the configuration.yaml in ROOT folder.
In configuration.yaml, you must change the path and name of your models and dataset.
Or you can test your model with command line like below command with your model and dataset path.
    python test.py --data your_yamlpath.yaml --model2_weight weight_path_of_model2 --model1_weight weight_path_of_model1 --model2_cfg cfg_path_of_model2

## Result check with my model
if you want to check my result with hackathon dataset, please follow the below direction.

1. please make the hackathon dataset in dataset folder.
2. Download the weights
  - yolov4 (https://drive.google.com/file/d/1XBls41JzDdbUC5SvPbUoahCXZ-KyLR-w/view?usp=sharing)
  - yolov5 (https://drive.google.com/file/d/1QhExYLCD8Wc4sf7XvWcIdy1zIK2B2j3k/view?usp=sharing)

and put the yolov4 weight in ROOT/model/yolov4/weights
put the yolov5 weight in ROOT/model/yolov5

3. implement the test code in yolov5 folder
   ```python
    cd ROOT
    python test.py --data dataset/custon.yaml --model2_weight model/yolov4/weights/v4_best.pt --model1_weight model/yolov5/v5_best.pt --model2_cfg model/yolov4/cfg/yolov4-pacsp-x.cfg
   ```
   

## Result
||Precision|Recall|$mAP<sub>0.5<sub>|$mAP<sub>0.5 - 0.95<sub>|
|---|---|---|---|---|
|YOLOv4|0.5261|0.8096|0.696|0.5838|
|YOLOv5|0.6018|0.7129|0.5344|0.6673|
|Ensemble|0.8158|0.8988|0.9192|0.8184|

## Reference
Weighted-boxes-fusion: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
  
Original paper: https://arxiv.org/abs/1910.13302

YOLOv4: https://github.com/WongKinYiu/PyTorch_YOLOv4

YOLOv5: https://github.com/ultralytics/yolov5
  
Thank you! If you have any question on my repo, please feel free to contact to me with your issue.
