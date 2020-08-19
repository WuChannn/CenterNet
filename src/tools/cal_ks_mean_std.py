import os
import cv2
import json
import numpy as np
from PIL import Image

def image_resize(image_file, new_path, new_w, new_h):
    image = Image.open(image_file)
    image = image.resize((new_w, new_h))
    image_name = image_file.split('dataset/')
    image.save(new_path + image_name[-1])

def get_image_from_json(json_file):
    image_files = []
    with open(json_file, 'r') as f:
        dst = json.load(f)
        images = dst['images']
        for image in images:
            image_files.append(image['file_name'])
        return image_files

if __name__ == '__main__':
    image_files = get_image_from_json('../data/ks/ks_train.json')
    new_path = '/root/data/ks/dataset/resized_train/'
    for image_file in image_files:
        image_resize(image_file, new_path, 512, 512)
    
    R_channel = 0
    G_channel = 0
    B_channel = 0
    
    for image_file in image_files:
        new_path = '/root/data/ks/dataset/resized_train/'
        image_name = image_file.split('dataset/')
        new_path = os.path.join(new_path, image_name[-1])
        image = cv2.imread(new_path) / 255.0
        R_channel = R_channel + np.sum(image[:, :, 0])
        G_channel = G_channel + np.sum(image[:, :, 1])
        B_channel = B_channel + np.sum(image[:, :, 2])
    
    num = len(image_files) * 512 * 512
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    
    for image_file in image_files:
        new_path = '/root/data/ks/dataset/resized_train/'
        image_name = image_file.split('dataset/')
        new_path = os.path.join(new_path, image_name[-1])
        image = cv2.imread(new_path) / 255.0
        R_channel = R_channel + np.sum((image[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((image[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((image[:, :, 2] - B_mean) ** 2)
    
    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)

    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))  




