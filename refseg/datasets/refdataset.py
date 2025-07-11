import copy
import logging
import random
from typing import Dict, List, Union, Mapping
import datasets
import mmcv
from mmengine import Config, list_from_file, print_log
from mmengine.dataset import Compose
from pycocotools import mask
from torch.utils.data import Dataset
from mmseg.datasets import LoadAnnotations
from mmseg.registry import DATASETS, TRANSFORMS



@DATASETS.register_module()
class RefSegDataset(Dataset):
    METAINFO = dict(
        classes=('background', 'mask'),
        palette=[[0, 0, 0], [255, 255, 255]])

    refsegrs_categories = [
        "road", "vehicle", "car", "van", "building", "truck",
        "trailer", "bus", "road marking", "bikeway", "sidewalk",
        "tree", "low vegetation", "impervious surface"
    ]
    risbench_categories =[
    "expressway service area",
    "expressway toll station",
    "ground track field",
    "basketball court",
    "container crane",
    "roundabout",
    "windmill",
    "overpass",
    "stadium",
    "bridge",
    "soccer ball field",
    "baseball diamond",
    "train station",
    "golf field",
    "airport",
    "harbor",
    "dam",
    "ship",
    "helipad",
    "vehicle",
    "chimney",
    "airplane",
    "helicopter",
    "tennis court",
    "storage tank",
    "swimming pool"
]
    def __init__(self,
                 data_root: str,
                 ann_file: str,  # jsonl file
                 pipeline: List[Dict] = None,
                 test_mode: bool = False,
                 reduce_zero_label: bool = False,
                 metainfo: Union[Mapping, Config, None] = None,
                 predict_mode: bool = False,
                 ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.dataset = datasets.load_dataset('json', data_files=ann_file)['train']  # always use the train split if loaded from jsonl
        if predict_mode:
            self.dataset = self.dataset.select(range(0, len(self.dataset), 20))
        # self.dataset = self.dataset.select(range(8))  # for debugging
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.reduce_zero_label = reduce_zero_label
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        if 'refsegrs' in self.ann_file:
            self.categories = self.refsegrs_categories
        elif 'risbench' in self.ann_file:
            self.categories = self.risbench_categories
        else:
            pass


    @classmethod
    def _load_metainfo(cls, metainfo: Union[Mapping, Config, None] = None) -> dict:
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, (Mapping, Config)):
            raise TypeError('metainfo should be a Mapping or Config, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo


    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def __len__(self):
        return len(self.dataset)

    def find_first_category(self, sentence):
        """
        Find which category from the given list appears first in the sentence.

        Args:
            sentence (str): The input sentence
            categories (list): List of category strings to check

        Returns:
            str: The first occurring category, or None if no category is found
        """
        sentence = sentence.lower()
        first_cat = None
        first_pos = float('inf')

        for category in self.categories:
            pos = sentence.find(category)
            if pos != -1 and pos < first_pos:
                first_pos = pos
                first_cat = category

        return first_cat

    def get_item(self, idx: int):
        item = self.dataset[idx]
        if isinstance(item['segmentation'], list):
            seg_mask = item['segmentation'][0]
        else:
            seg_mask = item['segmentation']
        results = dict()
        results['img_path'] = self.data_root + '/' + item['file_name']
        if item['file_name'] in ['22187.jpg', '20203.jpg', '00413.jpg']:
            print('Skipping image:', item['file_name'])
            return self.get_item(random.randint(0, len(self.dataset)))
        results['segmentation'] = seg_mask
        results['text'] = item['sent']
        results['reduce_zero_label'] = self.reduce_zero_label
        results['seg_fields'] = []

        # get category name
        if 'category_name' in item:
            results['category_name'] = item['category_name']
            assert results['category_name'] is not None, item['sent']
        # else:
        #     results['category_name'] = self.find_first_category(item['sent'])
        results = self.pipeline(results)
        return results


    def __getitem__(self, idx: int):
        try:
            return self.get_item(idx)
        except Exception as e:
            print('Error in RefSegDataset.__getitem__:', e)
            return self.get_item(random.randint(0, len(self.dataset)))



@TRANSFORMS.register_module()
class LoadSegAnnotations(LoadAnnotations):

    def _load_seg_map(self, results: dict) -> None:
        seg_mask_rle = results['segmentation']
        gt_semantic_seg = mask.decode(seg_mask_rle)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        mask_h, mask_w = gt_semantic_seg.shape
        img_h, img_w = results['ori_shape'][:2]
        if mask_w != img_w or mask_h != img_h:
            print('Resizing segmentation map... from', (mask_w, mask_h), 'to', (img_w, img_h))
            gt_semantic_seg = mmcv.imresize(gt_semantic_seg, (img_w, img_h), interpolation='nearest')
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
