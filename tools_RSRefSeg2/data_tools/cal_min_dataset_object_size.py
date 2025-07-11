import os

import cv2
import mmengine
import tqdm
from pycocotools import mask
import datasets


json_file = 'datainfo/rrsisd_test.jsonl'
data_infos = datasets.load_dataset('json', data_files=json_file)['train']
data_folder = '/mnt/ali-sh-1/dataset/cky_data/RRSIS-D/images/rrsisd/JPEGImages'

min_area = 10000
for item in tqdm.tqdm(data_infos):
    mask_rle = item['segmentation']
    img_file = item['file_name']
    if isinstance(mask_rle, list):
        mask_rle = mask_rle[0]
    mask_seg = mask.decode(mask_rle)
    area = mask_seg.sum()
    if area == 0:
        print('Zero area mask:', item['file_name'])
        print('Sentence:', item['sent'])
        print('bbox', item['bbox'])
        img = cv2.imread(data_folder + '/' + img_file)
        cv2.rectangle(img, (item['bbox'][0], item['bbox'][1]), (item['bbox'][2], item['bbox'][3]), (0, 0, 255), 2)
        cv2.imwrite('work_dirs/' + img_file, img)
        continue
    if area < min_area:
        min_area = area

print(min_area)
