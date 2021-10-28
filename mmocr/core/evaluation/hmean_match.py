# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmocr.utils as utils
from . import utils as eval_utils


def ignore_pred(pred_boxes, gt_ignored_index, gt_polys, precision_thr=0.8):
    """Ignore the predicted box if it hits any ignored ground truth.

    Args:
        pred_boxes (list[ndarray or list]): The predicted boxes of one image.
        gt_ignored_index (list[int]): The ignored ground truth index list.
        gt_polys (list[Polygon]): The polygon list of one image.
        precision_thr (float): The precision threshold.

    Returns:
        pred_polys (list[Polygon]): The predicted polygon list.
        pred_points (list[list]): The predicted box list represented
            by point sequences.
        pred_ignored_index (list[int]): The ignored text index list.
    """

    assert isinstance(pred_boxes, list)
    assert isinstance(gt_ignored_index, list)
    assert isinstance(gt_polys, list)
    assert 0 <= precision_thr <= 1

    pred_polys = []
    pred_points = []
    pred_ignored_index = []

    gt_ignored_num = len(gt_ignored_index)
    # get detection polygons
    for box_id, box in enumerate(pred_boxes):
        poly = eval_utils.points2polygon(box)
        pred_polys.append(poly)
        pred_points.append(box)

        if gt_ignored_num < 1:
            continue

        # ignore the current detection box
        # if its overlap with any ignored gt > precision_thr
        total_prec = 0.
        for ignored_box_id in gt_ignored_index:
            ignored_box = gt_polys[ignored_box_id]
            inter_area = eval_utils.poly_intersection(poly, ignored_box)
            area = poly.area
            precision = 0 if area == 0 else inter_area / area
            total_prec += precision
            if total_prec > precision_thr:
                pred_ignored_index.append(box_id)
                break

    return pred_polys, pred_points, pred_ignored_index


def cal_recall_or_prec_ratio(intersected_area, poly, reset=True):
    tmp_val = 1.0 * intersected_area / max(1, poly.area)
    if reset:
        min_x, min_y, max_x, max_y = poly.bounds
        w = max_x - min_x
        h = max_y - min_y
        h2w_ratio = 1.0 * max(w, h) / max(1, min(w, h))
        thresh = 1.0 / (int(h2w_ratio) + 1)
        # not a valid char(s) box
        if tmp_val < thresh:
            tmp_val = 0.0

    return tmp_val


def eval_hmean_match(pred_boxes,
                     gt_boxes,
                     gt_ignored_boxes,
                     perfect_match_thresh=0.8):
    """Evaluate hmean of text detection with real match.

    Args:
        pred_boxes (list[list[list[float]]]): Text boxes for an img list. Each
            box has 2k (>=8) values.
        gt_boxes (list[list[list[float]]]): Ground truth text boxes for an img
            list. Each box has 2k (>=8) values.
        gt_ignored_boxes (list[list[list[float]]]): Ignored ground truth text
            boxes for an img list. Each box has 2k (>=8) values.
        perfect_match_thresh (float): Match threshold when gt_box(es) and
            det_box(es) is matched perfectly.
    Returns:
        dataset_results (dict): Dict indicates the recall, precision,
            hmean of the dataset.
        img_results (list[dict]): Each dict indicates the recall,
            precision, hmean of one image.
    """
    assert utils.is_3dlist(pred_boxes)
    assert utils.is_3dlist(gt_boxes)
    assert utils.is_3dlist(gt_ignored_boxes)

    img_num = len(pred_boxes)
    assert img_num == len(gt_boxes)
    assert img_num == len(gt_ignored_boxes)

    dataset_gt_num = 0
    dataset_pred_num = 0
    dataset_recall, dataset_precision = 0., 0.

    img_results = []

    for i in range(img_num):
        gt = gt_boxes[i]
        gt_ignored = gt_ignored_boxes[i]
        pred = pred_boxes[i]

        gt_num = len(gt)
        gt_ignored_num = len(gt_ignored)
        pred_num = len(pred)

        recall, precision = 0., 0.

        # get gt polygons.
        gt_all = gt + gt_ignored
        gt_polys = [eval_utils.points2polygon(p) for p in gt_all]
        gt_ignored_index = [gt_num + i for i in range(len(gt_ignored))]
        gt_num = len(gt_polys)
        pred_polys, _, pred_ignored_index = ignore_pred(
            pred, gt_ignored_index, gt_polys, 0.8)

        if gt_num > 0 and pred_num > 0:
            sz = [gt_num, pred_num]
            recall_mat = np.zeros(sz)
            precision_mat = np.zeros(sz)
            # for recall and precision
            for gt_id in range(gt_num):
                for pred_id in range(pred_num):
                    gt_poly = gt_polys[gt_id]
                    det_poly = pred_polys[pred_id]
                    poly_inter = eval_utils.poly_intersection(
                        det_poly, gt_poly)
                    tmp_recall = cal_recall_or_prec_ratio(
                        poly_inter, gt_poly, reset=True)
                    tmp_prec = cal_recall_or_prec_ratio(
                        poly_inter, det_poly, reset=True)
                    recall_mat[gt_id, pred_id] = tmp_recall
                    precision_mat[gt_id, pred_id] = tmp_prec

            gt_hit = np.sum(recall_mat, axis=1, keepdims=False)
            pred_hit = np.sum(precision_mat, axis=0, keepdims=False)

            gt_care_number = gt_num - gt_ignored_num
            pred_care_number = pred_num - len(pred_ignored_index)

            # for recall
            for gt_id in range(gt_num):
                if gt_id in gt_ignored_index:
                    continue
                if gt_hit[gt_id] > perfect_match_thresh:
                    gt_hit[gt_id] = 1.0
                recall += gt_hit[gt_id]
            recall /= max(1, gt_care_number)

            # for precision
            for pred_id in range(pred_num):
                if pred_id in pred_ignored_index:
                    continue
                if pred_hit[pred_id] > perfect_match_thresh:
                    pred_hit[pred_id] = 1.0
                precision += pred_hit[pred_id]
            precision /= max(1, pred_care_number)

        denom = recall + precision
        hmean = 0.0 if denom == 0 else (2.0 * precision * recall / denom)

        img_results.append({
            'recall': recall,
            'precision': precision,
            'hmean': hmean
        })

        dataset_recall += (recall * gt_care_number)
        dataset_precision += (precision * pred_care_number)
        dataset_gt_num += gt_care_number
        dataset_pred_num += pred_care_number

    dataset_r = dataset_recall / max(1, dataset_gt_num)
    dataset_p = dataset_precision / max(1, dataset_pred_num)
    dataset_denom = dataset_r + dataset_p
    dataset_h = 0.0 if dataset_denom == 0 else (2.0 * dataset_p * dataset_r /
                                                dataset_denom)

    dataset_results = {
        'num_gts': dataset_gt_num,
        'num_dets': dataset_pred_num,
        'recall': dataset_r,
        'precision': dataset_p,
        'hmean': dataset_h
    }

    return dataset_results, img_results
