import json
import os
import numpy as np
import pycocotools.mask as mask_utils
import cv2
import mmengine

data_folder = '/mnt/ali-sh-1/dataset/cky_data/RefSegRS'
img_folder = 'images'
ann_folder = 'masks'


for split in ['train', 'val', 'test']:
	ann_text = os.path.join(data_folder, f'output_phrase_{split}.txt')
	ann_list = mmengine.list_from_file(ann_text)
	ann_list = [ann.strip() for ann in ann_list]
	ann_data = []
	for ann in ann_list:
		item = ann.split(' ')
		image_id = item[0]
		sent = ' '.join(item[1:])

		mask_path = os.path.join(data_folder, ann_folder, image_id+'.tif')
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		file_name = image_id+'.tif'
		# encode mask as segmentation using pycocotools
		segmentation = mask_utils.encode(np.asfortranarray(mask))
		segmentation = {
			'size': segmentation['size'],
			'counts': segmentation['counts'].decode('utf-8')
		}
		ann_data.append({
			'split': split,
			'image_id': image_id,
			'sent': sent,
			'file_name': file_name,
			'segmentation': segmentation,
		})
	print(f'split_set: {split}')
	os.makedirs('datainfo', exist_ok=True)
	with open(f'datainfo/refsegrs_{split}.jsonl', 'w') as f:
		for item in ann_data:
			if item['split'] == split:
				f.write(json.dumps(item) + '\n')