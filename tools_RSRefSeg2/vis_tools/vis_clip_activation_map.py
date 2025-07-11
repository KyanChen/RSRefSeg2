import glob
import json
import math
import os
import shutil
import PIL
import cv2
import einops
import mmengine
import numpy as np
import torch
import tqdm
from pycocotools import mask
from datasets import Dataset, DatasetDict
import datasets
import torch.nn.functional as F
import clip


def apply_overlay(activation, ori_img, save_folder, iou, filename, extra_key, colormap, alpha):
    w, h = ori_img.shape[:2]
    if isinstance(activation, torch.Tensor):
        activation = activation.detach().cpu().numpy()
    activation -= activation.min()
    activation /= (activation.max() + 1e-8)
    activation = np.uint8(255 * activation)
    # Resize to original image size
    activation_resized = cv2.resize(activation, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply colormap
    heatmap = cv2.applyColorMap(activation_resized, colormap)

    # Overlay heatmap onto image
    overlay = cv2.addWeighted(heatmap, alpha, ori_img, 1 - alpha, 0)
    cv2.imwrite(f'{save_folder}/{iou}_{filename}_{extra_key}.png', overlay)


def save_feat(item, save_folder, alpha, processor, model, colormap: int = cv2.COLORMAP_JET):
    item = item['json']
    iou, filename = os.path.basename(item).split('_')
    filename = filename.split('.')[0]
    ori_img = cv2.imread(item.replace('.json', '_img.png'))


    img = ori_img.copy()
    text = json.load(open(item))['sent']
    texts = [text, '', 'An image', 'A photo', 'A photo of remote sensing image', 'A remote sensing image', 'An', 'A', 'The', 'the']

    img = PIL.Image.fromarray(img)
    img = processor(img).to('cuda').unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(model, texts, 'cuda', prompt_templates=['{}'])
        similarity = clip.clip_feature_surgery(
            image_features,
            text_features[:1, :],
            redundant_feats=text_features[1:, :].mean(0, keepdim=True)
        )
        act = similarity[0, 1:, 0]
        act = einops.rearrange(act, '(h w) -> h w', h=int(act.shape[0] ** 0.5))
    apply_overlay(act, ori_img, save_folder, iou, filename, extra_key=f'clip_act', colormap=colormap, alpha=alpha)
    shutil.copy(item.replace('.json', '_img.png'), f'{save_folder}/{iou}_{filename}_img.png')
    shutil.copy(item, f'{save_folder}/{iou}_{filename}.json')
    shutil.copy(item.replace('.json', '_pred.png'), f'{save_folder}/{iou}_{filename}_pred.png')
    shutil.copy(item.replace('.json', '_gt.png'), f'{save_folder}/{iou}_{filename}_gt.png')



if __name__ == '__main__':
    data_root = 'work_dirs/visualizer/vis_data'
    save_folder = 'work_dirs/visualizer/vis_data/vis_clip'
    alpha = 0.6
    os.makedirs(save_folder, exist_ok=True)

    model, preprocess = clip.load("CS-ViT-B/16", device="cuda")
    model.eval()

    data_list = glob.glob(os.path.join(data_root, '*.json'))
    data_list.sort()
    import random
    random.shuffle(data_list)
    data_list = data_list
    print(len(data_list))
    dataset = Dataset.from_dict({'json': data_list})
    dataset.map(save_feat, num_proc=1, fn_kwargs={'save_folder': save_folder, 'alpha': alpha, 'processor': preprocess, 'model': model})


