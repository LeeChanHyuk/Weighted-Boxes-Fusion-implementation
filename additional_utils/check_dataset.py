import os
import cv2
from tqdm import tqdm
import json

test_image_path = '/media/ddl/새 볼륨/Git/hackathon_project/dataset/9_쓰레기봉투/9_쓰레기봉투'
test_json_path = '/media/ddl/새 볼륨/Git/hackathon_project/dataset/9_쓰레기봉투/9_쓰레기봉투'

def check_annotation(image_path, json_path):
    for image_name in os.listdir(image_path):
        print(image_name.split('.'))
        with open(os.path.join(json_path,image_name.split('.')[0] + ".json"), "r") as json_file:
            json_dict = json.load(json_file)
            image = cv2.imread(os.path.join(image_path, image_name))
            coordinate, length = json_dict['annotations'][0]['bbox']
            x, y, width, height = coordinate[0], coordinate[1], length[0], length[1]
            image = cv2.rectangle(image, (int(x), int(y)), (int(width), int(height)), (0, 0, 255), 3)
            cv2.imshow('image', image)
            cv2.waitKey(0)

base_data_path = '/home/ddl/다운로드/zip/dataset'
for folder in os.listdir(base_data_path):
    file_list = os.listdir(os.path.join(base_data_path, folder, folder))
    for file in file_list:
        file_prefix = file.split('.')[0]
        if (file_prefix + '.json' in file_list) and (file_prefix + '.jpg' in file_list):
            a = 1
        else:
            try:
                os.remove(os.path.join(base_data_path, folder, folder, file_prefix + '.json'))
            except:
                print(file_prefix + '.json can not deleted' )
            try:
                os.remove(os.path.join(base_data_path, folder, folder, file_prefix + '.jpg'))
            except:
                print(file_prefix + '.jpg can not deleted' )
