# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# For using yolov5 libraries
sys.path.append('/home/ddl/git/WBF/Weighted-Boxes-Fusion-implementation2/model/yolov5')
from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.callbacks import Callbacks
from model.yolov5.utils.datasets import create_dataloader
from model.yolov5.utils.general import (LOGGER, box_iou, check_dataset as check, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path as increment, non_max_suppression as non_maximum_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from model.yolov5.utils.metrics import ConfusionMatrix, ap_per_class
from model.yolov5.utils.plots import output_to_target, plot_images, plot_val_study
from model.yolov5.utils.torch_utils import select_device, time_sync

from model.yolov5.utils.general import (
    check_img_size, non_max_suppression,  scale_coords, xyxy2xywh, strip_optimizer)
from model.yolov5.utils.torch_utils import select_device
from model.yolov5.utils.datasets import *
from model.yolov5.utils.general import *

from model.yolov5.WBF.examples.example import example_wbf_2_models, example_wbf_1_model
from model.yolov4.models.models import *

import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def parse_yaml(file_path, encoding="utf-8"):
    assert os.path.isfile(file_path) 
    with open(file_path, "r", encoding=encoding) as f:
        dict_yaml = yaml.load(f, Loader=yaml.FullLoader)
        return dict_yaml


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='test',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/test',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model2_weight = 'default',
        model2_cfg = 'default',
        model1_weight = 'default',
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    test_configuration = parse_yaml(ROOT/'configuration.yaml')
    data = os.path.join(test_configuration['dataset']['configuration_path'], test_configuration['dataset']['configuration_name'])

    model1_weight_path = os.path.join(test_configuration['model1']['weight_folder_path'], test_configuration['model1']['weight_file_name'])
    if test_configuration['model1']['cfg_file']:
        model1_cfg_path = os.path.join(test_configuration['model1']['cfg_folder_path'], test_configuration['model1']['cfg_file_name'])

    model2_weight_path = os.path.join(test_configuration['model2']['weight_folder_path'], test_configuration['model2']['weight_file_name'])
    if test_configuration['model2']['cfg_file']:
        model2_cfg_path = os.path.join(test_configuration['model2']['cfg_folder_path'], test_configuration['model2']['cfg_file_name'])
    
    # Initialize/load model and set device
    if opt.data != 'default':
        data = str(opt.data)
    if opt.model1_weight != 'default':
        model1_weights_path = str(opt.model1_weight)

    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


        # Load VOLOv5 model (Model 1)
        #v5_model = DetectMultiBackend(weights, device=device, dnn=dnn)
        model1 = DetectMultiBackend(model1_weight_path, device=device, dnn=dnn)
        stride, pt, jit, engine = model1.stride, model1.pt, model1.jit, model1.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model1.model.half() if half else model1.model.float()
        elif engine:
            batch_size = model1.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Load VOLOv4 model
        if opt.model2_cfg != 'default':
            model2_cfg_path = str(opt.model2_cfg)
        if opt.model2_weight != 'default':
            model2_weight_path = str(opt.model2_weight)
        model2 = Darknet(model2_cfg_path, imgsz).cuda()
        model2.load_state_dict(torch.load(model2_weight_path, map_location=device)['model'])
        model2.to(device).eval()
        if half:
            model2.half()  # to FP16

        # Data
        data = check(data)  # check

    # Configure
    model1.eval()
    model2.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        model1.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        task = 'test'
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model1.names if hasattr(model1, 'names') else model1.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        imgs=[]
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        model1_out = model1(im, augment=augment, val=False)  # inference, loss outputs
        model2_out = model2(im, augment=opt.augment)[0]
        dt[1] += time_sync() - t2

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        model1_out = non_maximum_suppression(model1_out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        model2_out = non_maximum_suppression(model2_out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        for si, (im0, model2_dets, model1_dets) in enumerate(zip(im, model1_out, model2_out)):
            im0 = im0.detach().cpu().numpy() * 255
            im0 = im0.transpose((1,2,0)).astype(np.uint8).copy()
            # concat two model's outputs
            if len(model2_dets):
                model2_dets[:, :4] = scale_coords(im.shape[2:], model2_dets[:, :4], im0.shape).round()

            if len(model1_dets):
                model1_dets[:, :4] = scale_coords(im.shape[2:], model1_dets[:, :4], im0.shape).round()
            
            # Flag for indicating detection success
            detect_success = False
            
            if len(model2_dets)>0 and len(model2_dets)>0:
                boxes, scores, labels = example_wbf_2_models(model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                for box in boxes:
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
                detect_success = True
            elif len(model2_dets)>0:
                boxes, scores, labels = example_wbf_1_model(model2_dets.detach().cpu().numpy(), im0)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                for box in boxes:
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
                detect_success = True
            elif len(model1_dets)>0:
                boxes, scores, labels = example_wbf_1_model(model1_dets.detach().cpu().numpy(), im0)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                for box in boxes:
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
                detect_success = True
            p_boxes, p_scores, p_labels = boxes, scores, labels
            # Result visualization
            #if detect_success is True:
                #cv2.imshow("detected_image", im0)
                #cv2.waitKey(0)
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(boxes) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            pred = np.concatenate([p_boxes, np.expand_dims(p_scores, axis=1), np.expand_dims(p_labels, axis=1)], axis=1)
            pred = torch.from_numpy(pred).to(device)
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='default', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='default', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'model/yolov5/runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--model2_weight', nargs='+', type=str, default='default')
    parser.add_argument('--model2_cfg', nargs='+', type=str, default='default')
    parser.add_argument('--model1_weight', nargs='+', type=str, default='default')
    opt = parser.parse_args()
    #opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        result, maps, t = run(**vars(opt))
        print('Map 0.5 is ', result[2],'Map 0.5 to 0.95 is', result[3])


    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
