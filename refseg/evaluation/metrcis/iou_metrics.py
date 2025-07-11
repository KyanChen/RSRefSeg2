from typing import Optional, Sequence, Dict
import torch
from mmengine import mkdir_or_exist, MMLogger, print_log
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
from mmseg.registry import METRICS


@METRICS.register_module()
class RefSegIoUMetric(BaseMetric):
    default_prefix = 'RefSeg'
    def __init__(self,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            if 'category_name' in data_sample:
                category_name = data_sample['category_name']
            else:
                category_name = 'none'
            self.results.append((self.intersect_and_union(pred_label, label, num_classes, self.ignore_index), category_name))

    def compute_per_class_iou_by_image(self, category_names, total_area_intersect, total_area_union):
        """
        按图像计算每个类别的 IoU，再对同一类别所有图像的 IoU 求平均。

        Args:
            category_names (list of str): 每张图像对应的类别名称列表，长度为 N。
            total_area_intersect (list of float): 每张图像预测与真实的交集面积，长度为 N。
            total_area_union (list of float): 每张图像预测与真实的并集面积，长度为 N.

        Returns:
            class_iou_dict (dict): {类别名: 平均 IoU}，即该类别在各图像上的 IoU 平均值。
            mean_iou (float): 所有类别平均 IoU 的算术平均值（mIoU）。
        """
        from collections import defaultdict

        # 收集每张图像在该类别上的 IoU
        iou_list_per_class = defaultdict(list)
        # import ipdb; ipdb.set_trace()
        for cls, inter, uni in zip(category_names, total_area_intersect, total_area_union):
            inter = inter[-1].item()
            uni = uni[-1].item()
            if uni > 0:
                iou = inter / uni
            else:
                iou = float('nan')
            iou_list_per_class[cls].append(iou)

        # 计算每个类别的平均 IoU
        class_iou_dict = {}
        for cls, iou_list in iou_list_per_class.items():
            # 过滤掉 nan
            valid = [x for x in iou_list if x == x]
            if valid:
                class_iou_dict[cls] = sum(valid)*100 / len(valid)
            else:
                class_iou_dict[cls] = float('nan')

        # 计算整体 mIoU
        valid_class_ious = [v for v in class_iou_dict.values() if v == v]
        mean_iou = sum(valid_class_ious) / len(valid_class_ious) if valid_class_ious else float('nan')

        return class_iou_dict, mean_iou


    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])

        category_names = [x[-1] for x in results]
        results = [x[0] for x in results]
        results = tuple(zip(*results))
        assert len(results) == 4


        total_area_intersect, total_area_union, total_area_pred_label, total_area_label = results
        return_metrics = dict()

        class_iou, miou = self.compute_per_class_iou_by_image(category_names, total_area_intersect, total_area_union)
        print("Per-class Average IoU (by image):")
        for cls, iou in class_iou.items():
            print(f"  {cls}: {iou:.2f}")
            return_metrics[cls] = iou
        print(f"Mean IoU (mIoU): {miou:.2f}")
        return_metrics['mean_iou'] = miou

        # calculate the cumulative IoU and the generalized IoU
        cIoU = sum(total_area_intersect) / sum(total_area_union)
        per_image_iou = [area_intersect / area_union for area_intersect, area_union in zip(total_area_intersect, total_area_union)]
        gIoU = sum(per_image_iou) / len(per_image_iou)

        for id_cls in range(len(cIoU)):
            return_metrics[f'cIoU_{id_cls}'] = cIoU[id_cls].item()*100
            return_metrics[f'gIoU_{id_cls}'] = gIoU[id_cls].item()*100


        # calculate the segmentation accuracy for each IoU threshold (0.5, 0.6, 0.7, 0.8, 0.9)
        seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
        per_image_iou = torch.stack(per_image_iou)  # BxNC
        for i, iou in enumerate(seg_iou_list):
            seg_correct = (per_image_iou > iou).sum(dim=0).float() / len(per_image_iou)
            for id_cls, correct in enumerate(seg_correct):
                return_metrics[f'seg_acc_{iou}_{id_cls}'] = correct.item()*100

        class_table_data = PrettyTable()
        class_table_data.field_names = ['class', 'cIoU', 'gIoU'] + [f'seg_acc_{iou}' for iou in seg_iou_list]
        for id_cls, class_name in enumerate(self.dataset_meta['classes']):
            class_table_data.add_row([class_name, round(return_metrics[f'cIoU_{id_cls}'], 2), round(return_metrics[f'gIoU_{id_cls}'], 2)] + [round(return_metrics[f'seg_acc_{iou}_{id_cls}'], 2) for iou in seg_iou_list])


        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return return_metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor, num_classes: int, ignore_index: int):
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label