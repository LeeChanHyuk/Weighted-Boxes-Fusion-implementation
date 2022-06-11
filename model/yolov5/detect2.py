# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
import platform
import shutil

import cv2
import torch
import torch.backends.cudnn as cudnn
from copy import deepcopy
from numpy import mat, random, save
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
sys.path.append('/home/ddl/git/hackathon_project/concat')
sys.path.append('/home/ddl/git/hackathon_project/concat/yolov5-master')
sys.path.append('/home/ddl/git/hackathon_project/concat/yolov5-master/PyTorch_YOLOv4-master')
sys.path.append('/home/ddl/git/hackathon_project/concat/yolov5/WBF/examples')
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


from yolov4.v4_utils.google_utils import attempt_load as v4_attempt_load
from utils.general import (
    check_img_size, non_max_suppression,  scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device
from yolov4.models.models import *
from utils.datasets import *
from utils.general import *

from WBF.examples.example import example_wbf_2_models, example_wbf_1_model

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

@torch.no_grad()
def run(weights=ROOT / 'v5_best.pt',  # model.pt path(s)
        source='/home/ddl/git/hackathon_project/dataset/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        cfg = '',
        v4_weights='',
        save_path = ''
        ):
    save_path = str(opt.save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    weights = str(opt.weights)
    source = str(opt.source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load YOLOv5
    device = select_device(device)
    v5_model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = v5_model.stride, v5_model.names, v5_model.pt, v5_model.jit, v5_model.onnx, v5_model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load YOLOv4
    cfg = str(opt.cfg)
    v4_weights = str(opt.v4_weights)
    v4_model = Darknet(cfg, imgsz).cuda()
    try:
        v4_model.load_state_dict(torch.load(v4_weights, map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(v4_model, v4_weights)
    v4_model.to(device).eval()
    if half:
        v4_model.half()  # to FP16

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        v5_model.model.half() if half else v5_model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    v5_model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in tqdm(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        v5_pred = v5_model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        v4_pred = v4_model(im, augment=opt.augment)[0]

        # NMS
        v5_pred = non_max_suppression(v5_pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        v4_pred = non_max_suppression(v4_pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        color_list = []
        for i in range(13): # class number
            color_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        for i, (v4_dets, v5_dets) in enumerate(zip(v4_pred, v5_pred)): # for each image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            image_for_yolo = deepcopy(im0s)

            if len(v4_dets):
                v4_dets[:, :4] = scale_coords(im.shape[2:], v4_dets[:, :4], im0.shape).round()

            if len(v5_dets):
                v5_dets[:, :4] = scale_coords(im.shape[2:], v5_dets[:, :4], im0.shape).round()
            
            # Flag for indicating detection success
            detect_success = False
            
            if len(im0.shape)==3:
                img_height, img_width = im0.shape[0:2]
            elif len(im0.shape) == 4:
                img_height, img_width = im0.shape[1:3]
                
            if len(v4_dets)>0 and len(v5_dets)>0:
                boxes, scores, labels = example_wbf_2_models(v4_dets.detach().cpu().numpy(), v5_dets.detach().cpu().numpy(), im0)
                boxes[:,0], boxes[:,2] = boxes[:,0] * img_width, boxes[:,2] * img_width
                boxes[:,1], boxes[:,3] = boxes[:,1] * img_height, boxes[:,3] * img_height
                for i, box in enumerate(boxes):
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_list[int(labels[i])], 3)
                    cv2.putText(im0, 'class-'+str(int(labels[i])),(int(box[0]), int(box[1])), 3,1, color_list[int(labels[i])], 2,cv2.LINE_AA)
                detect_success = True
            elif len(v4_dets)>0:
                boxes, scores, labels = example_wbf_1_model(v4_dets.detach().cpu().numpy(), im0)
                boxes[:,0], boxes[:,2] = boxes[:,0] * img_width, boxes[:,2] * img_width
                boxes[:,1], boxes[:,3] = boxes[:,1] * img_height, boxes[:,3] * img_height
                for i, box in enumerate(boxes):
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_list[int(labels[i])], 3)
                    cv2.putText(im0, 'class-'+str(int(labels[i])),(int(box[0]), int(box[1])), 3,1, color_list[int(labels[i])], 2,cv2.LINE_AA)
                detect_success = True
            elif len(v5_dets)>0:
                boxes, scores, labels = example_wbf_1_model(v5_dets.detach().cpu().numpy(), im0)
                boxes[:,0], boxes[:,2] = boxes[:,0] * img_width, boxes[:,2] * img_width
                boxes[:,1], boxes[:,3] = boxes[:,1] * img_height, boxes[:,3] * img_height
                for i, box in enumerate(boxes):
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_list[int(labels[i])], 3)
                    cv2.putText(im0, 'class-'+str(int(labels[i])),(int(box[0]), int(box[1])), 3,1, color_list[int(labels[i])], 2,cv2.LINE_AA)
                detect_success = True

                

            # Result visualization
            if detect_success is True:
                # Showing result
                #cv2.imshow("detected_image", im0)
                #cv2.waitKey(0)
                # save the result
                file_name = path.split('/')[-1]
                cv2.imwrite(os.path.join(save_path, file_name), im0)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'v5_best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '/home/ddl/git/hackathon_project/dataset/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--cfg', nargs='+', type=str, default=ROOT / 'yolov4/cfg/yolov4-pacsp-x.cfg')
    parser.add_argument('--v4_weights', nargs='+', type=str, default=ROOT / 'yolov4/weights/v4_best.pt')
    parser.add_argument('--save_path', nargs='+', type=str, default= ROOT / 'DETECTION_RESULT')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)

    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
