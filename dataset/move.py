import os
import shutil

original_path = '/home/ddl/git/hackathon_project/dataset/images'
original_path2 = '/home/ddl/git/hackathon_project/dataset/annotations'
save_path = '/home/ddl/git/hackathon_project/dataset/whole_data'

for i in os.listdir(original_path):
    shutil.copy(os.path.join(original_path, i), os.path.join(save_path, i))
for i in os.listdir(original_path2):
    shutil.copy(os.path.join(original_path2, i), os.path.join(save_path, i))