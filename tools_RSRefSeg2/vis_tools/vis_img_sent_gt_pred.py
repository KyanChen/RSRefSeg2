import glob
import os
import cv2
import mmengine
import numpy as np
import tqdm
from pycocotools import mask
import datasets


def compute_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(intersection) / union if union > 0 else 0.0


def save_img_sent_gt_pred(item, pred_folder, save_folder, alpha):
    if isinstance(item['segmentation'], list):
        seg_mask = item['segmentation'][0]
    else:
        seg_mask = item['segmentation']
    img_name = item['file_name'].split('.')[0]
    img_path = data_root + '/' + item['file_name']

    img = cv2.imread(img_path)
    seg_mask = mask.decode(seg_mask)
    gt = cv2.resize(seg_mask, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    if not os.path.exists(pred_folder + '/test_' + item['file_name'].split('.')[0] + '.png'):
        return
    pred = cv2.imread(pred_folder + '/test_' + item['file_name'].split('.')[0] + '.png')
    pred_w = pred.shape[1]
    pred = pred[:, pred_w//2:]
    pred = pred[:, :, 0]
    pred = cv2.resize(pred, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    # save IoU and sent
    sent = item['sent']
    IoU = compute_iou(gt, pred == 255)
    iou_str = '{}'.format(int(IoU*10000))
    mmengine.dump({
        'sent': sent,
        'IoU': IoU,
    }, save_folder + '/' + iou_str + '_' + img_name + '.json')

    # save_img
    cv2.imwrite(save_folder + '/' + iou_str + '_' + img_name + '_img.png', img)

    # save gt
    mask_img = img.copy()
    # mask_img[gt == 1] = mask_img[gt == 1] * (1 - alpha) + alpha * np.array([0, 0, 255])
    mask_img = mask_img * (1 - alpha) + alpha * gt[:, :, None] * np.array([0, 0, 255])
    cv2.imwrite(save_folder + '/' + iou_str + '_' + img_name + '_gt.png', mask_img)

    # save pred
    pred_img = img.copy()
    # pred_img[pred == 255] = pred_img[pred == 255] * (1 - alpha) + alpha * np.array([0, 0, 255])
    pred_img = pred_img * (1 - alpha) + alpha * (pred == 255)[:, :, None] * np.array([0, 0, 255])
    cv2.imwrite(save_folder + '/' + iou_str + '_' + img_name + '_pred.png', pred_img)


if __name__ == '__main__':
    # You should first run the inference to generate the prediction images in the pred_folder.


    # data_root = '/mnt/ali-sh-1/dataset/cky_data/RefSegRS/images/'
    # dataset = datasets.load_dataset('json', data_files='datainfo/refsegrs_test.jsonl')['train']
    # pred_folder = 'work_dirs/refsegrs_vis/visualizer/vis_data/vis_image'
    # save_folder = 'work_dirs/refsegrs_vis/visualizer/vis_data/vis_paper'

    data_root = '/mnt/ali-sh-1/dataset/cky_data/RRSIS-D/images/rrsisd/JPEGImages/'
    dataset = datasets.load_dataset('json', data_files='datainfo/rrsisd_test.jsonl')['train']
    pred_folder = 'work_dirs/rrsisd_vis/visualizer/vis_data/vis_image'
    save_folder = 'work_dirs/rrsisd_vis/visualizer/vis_data/vis_paper'

    # data_root = '/mnt/ali-sh-1/dataset/cky_data/RISBench_dataset/img_rgb'
    # dataset = datasets.load_dataset('json', data_files='datainfo/risbench_test.jsonl')['train']
    # pred_folder = 'work_dirs/risbench_vis/visualizer/vis_data/vis_image'
    # save_folder = 'work_dirs/risbench_vis/visualizer/vis_data/vis_paper'
    alpha = 0.6


    os.makedirs(save_folder, exist_ok=True)

    pred_list = glob.glob(os.path.join(pred_folder, '*.png'))
    for pred_path in tqdm.tqdm(pred_list):
        os.rename(pred_path, pred_folder + '/' + os.path.basename(pred_path).split('.')[0] + '.png')


    dataset.map(save_img_sent_gt_pred, num_proc=32, fn_kwargs={'pred_folder': pred_folder, 'save_folder': save_folder, 'alpha': alpha})


