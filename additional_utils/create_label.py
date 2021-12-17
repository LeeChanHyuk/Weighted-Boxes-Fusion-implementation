import os
import cv2
from numpy.lib.npyio import save
from tqdm import tqdm
import json
import numpy as np

## this file is working when the json file is splitted with each images. (ex) 1.jpg, 1.json, 2.jpg, 2.json ...
## please move all the files in one folder (you must do this!)

data_path = '/home/ddl/다운로드/zip/dataset2/1'
save_path = '/content/drive/MyDrive/illegal_object_detection/PyTorch_YOLOv4/dataset/whole_data'

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1004)
split2 = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=1004)

# check pair
def check_pair(data_path):
    file_list = os.listdir(data_path)
    for file in file_list:
        file_prefix = file.split('.')[0]
        if (file_prefix + '.json' in file_list) and (file_prefix + '.jpg' in file_list):
            a = 1
        else:
            try:
                os.remove(os.path.join(data_path, file_prefix + '.json'))
            except:
                print(file_prefix + '.json can not deleted' )
            try:
                os.remove(os.path.join(data_path, file_prefix + '.jpg'))
            except:
                print(file_prefix + '.jpg can not deleted' )

check_pair(data_path)

## Extract file name with each class number
image_list = []
label_list = []
json_list = []
for file in os.listdir(data_path):
    if file[len(file)-3:] == 'jpg':
        image_list.append(file)
        with open(os.path.join(data_path,file.split('.')[0] + ".json"), "r") as json_file:
            json_dict = json.load(json_file)
            json_list.append(file.split('.')[0] + '.json')
            class_num = json_dict['category'][0]['id']
            label_list.append(class_num)

# make txt file from data and json
def make_data_label_file(xs, jsons, txtname, base_path, save_path):
    f = open(os.path.join(txtname), 'w')
    for i in range(len(xs)):
        image_name = xs[i]
        json_name = jsons[i]
        f.write(os.path.join(save_path, image_name+' '))

        # json parsing
        with open(os.path.join(base_path, json_name), "r") as json_file:
            json_dict = json.load(json_file)
            for i in range(len(json_dict['category'])):
                class_num = json_dict['category'][i]['id']
                image_height, image_width = json_dict['images']['height'], json_dict['images']['width']
                coordinate, length = json_dict['annotations'][i]['bbox']
                x1, y1, width, height = coordinate[0], coordinate[1], length[0], length[1]
                center_x, center_y = x1 + (width/2), y1 + (height/2)
                x, y, w, h = center_x / image_width, center_y / image_height, width / image_width, height / image_height
                f.write(str(class_num)+','+str(x)+','+str(y)+','+str(w)+','+str(h)+' ')
            f.write('\n')
    f.close()

# change class number for kfold (if your class number start with 0 to sort of class, you don't have to do this)
label_list = [0 if value==9 else value for value in label_list]
label_list = [1 if value==13 else value for value in label_list]
label_list = [2 if value==14 else value for value in label_list]
label_list = [3 if value==17 else value for value in label_list]
label_list = [4 if value==18 else value for value in label_list]
label_list = [5 if value==19 else value for value in label_list]
label_list = [6 if value==20 else value for value in label_list]
label_list = [7 if value==21 else value for value in label_list]
label_list = [8 if value==22 else value for value in label_list]
label_list = [9 if value==23 else value for value in label_list]
label_list = [10 if value==24 else value for value in label_list]
label_list = [11 if value==25 else value for value in label_list]
label_list = [12 if value==26 else value for value in label_list]

## K-Fold for making train, valid, test dataset (txt file)
for train_idx, test_idx in split.split(image_list, np.array(label_list)):
    train_idx = train_idx.tolist()
    test_idx = test_idx.tolist()
    x_train = []
    y_train = []
    json_train = []
    x_test_split = []
    y_test_split = []
    json_test_split = []
    for i in train_idx:
        x_train.append(image_list[i])
        y_train.append(label_list[i])
        json_train.append(json_list[i])
    for i in test_idx:
        x_test_split.append(image_list[i])
        y_test_split.append(label_list[i])
        json_test_split.append(json_list[i])

    for index, (train_split_idx, valid_split_idx) in enumerate(split2.split(x_train, np.array(y_train))):
        train_split_idx = train_split_idx.tolist()
        valid_split_idx = valid_split_idx.tolist()
        x_train_split = []
        y_train_split = []
        json_train_split = []
        x_valid_split = []
        y_valid_split = []
        json_valid_split = []
        for i in train_split_idx:
            x_train_split.append(x_train[i])
            json_train_split.append(json_train[i])
        for i in valid_split_idx:
            x_valid_split.append(x_train[i])
            json_valid_split.append(json_train[i])
        make_data_label_file(x_train_split, json_train_split, 'train_' + str(index) + '.txt', data_path, save_path)
        make_data_label_file(x_valid_split, json_valid_split, 'valid_' + str(index) + '.txt', data_path, save_path)
    make_data_label_file(x_train_split, json_train_split, 'test_' + str(0) + '.txt', data_path, save_path)
    break

