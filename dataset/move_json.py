import os
import json
import cv2

data_path = '/home/ddl/git/hackathon_project/dataset/whole_data'
image_path = '/home/ddl/git/hackathon_project/dataset/images'
json_path = '/home/ddl/git/hackathon_project/dataset/annotations'
txt_path = '/home/ddl/git/hackathon_project/dataset/labels'


def move_file(data_path, image_path, json_path):
    for file in os.listdir(data_path):
        if file[len(file)-3:] == 'jpg':
            os.rename(os.path.join(data_path, file), os.path.join(image_path, file))
        if file[len(file)-4:] == 'json':
            os.rename(os.path.join(data_path, file), os.path.join(json_path, file))

# make txt file from data and json
def make_data_label_file(jsons, base_path, save_path):
    count = 0
    for i in range(len(jsons)):
        txtname = jsons[i].split('.')[0] + '.txt'
        f = open(os.path.join(save_path, txtname), 'w')
        json_name = jsons[i]
        image_name = json_name[:len(json_name)-4] + 'jpg'
        # json parsing
        with open(os.path.join(base_path, json_name), "r") as json_file:
            json_dict = json.load(json_file)
            for i in range(len(json_dict['category'])):
                class_num = json_dict['category'][i]['id']
                if class_num == 9:
                    class_num = 0
                elif class_num == 13:
                    class_num = 1
                elif class_num == 14:
                    class_num = 2
                elif class_num == 17:
                    class_num = 3
                elif class_num == 18:
                    class_num = 4
                elif class_num == 19:
                    class_num = 5
                elif class_num == 20:
                    class_num = 6
                elif class_num == 21:
                    class_num = 7
                elif class_num == 22:
                    class_num = 8
                elif class_num == 23:
                    class_num = 9
                elif class_num == 24:
                    class_num = 10
                elif class_num == 25:
                    class_num = 11
                elif class_num == 26:
                    class_num = 12
                else:
                    continue
                image_height, image_width = json_dict['images']['height'], json_dict['images']['width']
                coordinate, coordinate2 = json_dict['annotations'][i]['bbox']
                x1, y1, x2, y2 = coordinate[0], coordinate[1], coordinate2[0], coordinate2[1]
                center_x, center_y, width, height = (x1 + x2) / 2, (y1 + y2) / 2, x2-x1, y2-y1
                x, y, w, h = center_x / image_width, center_y / image_height, width / image_width, height / image_height
                f.write(str(class_num)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')

jsons = os.listdir(json_path)
make_data_label_file(jsons, json_path, txt_path)