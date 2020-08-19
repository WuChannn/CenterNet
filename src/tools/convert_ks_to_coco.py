import cv2
import json
import numpy as np
from tqdm import tqdm

classname_to_id = {'shu':1, 'tuo':2, 'fang':3, 'ks':4, 'other':5, 'frac':6}

class convert_label_to_coco_format(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)

    def load_annotations(self):
        if self.is_train:
            annot_path = '/root/data/ks/code/CenterNet_xing/data/ks/ks_train.txt'
        else:
            annot_path = '/root/data/ks/code/CenterNet_xing/data/ks/ks_test.txt'
        with open(annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            annotations = [{'img': i.split()[0], 'coor': i} for i in annotations]
        np.random.shuffle(annotations)
        return annotations

    def parse_annotation(self, annotation):

        image = annotation['img']
        line = annotation['coor']

        line = line.split()

        bboxes = line[1:]
        bboxes = [i.split(',') for i in bboxes]
        bboxes = np.array(bboxes, dtype=int)

        return image, bboxes

    def to_coco(self, save_path):
        self._init_categories()
        annotations = self.load_annotations()
        pbar = tqdm(annotations)
        for annotation in pbar:
            image, bboxes = self.parse_annotation(annotation)
            self.images.append(self._image(image))
            
            for bbox in bboxes:
                self.annotations.append(self._annotation(bbox))
                self.ann_id += 1
            self.img_id += 1

        instance = {}
        instance['info'] = 'wzc created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        # print(instance)
        self.save_coco_json(instance, save_path)
        # return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        # print(path)
        img = cv2.imread(path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, bbox):
        point = bbox[:4]
        label = bbox[4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(label) + 1
        annotation['segmentation'] = self._get_seg(point)
        annotation['bbox'] = [int(point[0]), int(point[1]), int(point[2]), int(point[3])]
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(point)
        return annotation

    # 计算面积
    def _get_area(self, point):
        min_x = point[0]
        min_y = point[1]
        max_x = point[2]
        max_y = point[3]
        return int((max_x - min_x+1) * (max_y - min_y+1))

    # segmentation
    def _get_seg(self, point):
        min_x = int(point[0])
        min_y = int(point[1])
        max_x = int(point[2])
        max_y = int(point[3])
        h = max_y - min_y
        w = max_x - min_x

        a = [min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y]
        return a

if __name__ == '__main__':
    train_labels = convert_label_to_coco_format(is_train=True)
    train_labels.to_coco('/root/data/ks/code/CenterNet_xing/data/ks/ks_train.json')
    train_labels = convert_label_to_coco_format(is_train=False)
    train_labels.to_coco('/root/data/ks/code/CenterNet_xing/data/ks/ks_test.json')