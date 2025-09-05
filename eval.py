import re
import time
import os
import torch
import numpy as np
from skimage.measure import regionprops
from torch.nn import functional as F
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.metrics import auc

from utils import t2np, rescale
from functools import partial
from multiprocessing import Pool
from skimage.measure import label, regionprops
from visualization import (
    save_anomaly_visualization,
)

from UniNet_lib.mechanism import weighted_decision_mechanism


def evaluation_indusAD(
    c, model, dataloader, device, save_visuals=False
):  # Changed is_train to save_visuals
    model.train_or_eval(type="eval")
    n = model.n

    # Lists to store original images and their paths for visualization
    original_images = []
    original_paths = []
    gt_list_px = []
    gt_list_sp = []
    output_list = [list() for _ in range(n * 3)]
    weights_cnt = 0

    start_time = time.time()
    with torch.no_grad():
        for idx, (sample, label, gt, path) in enumerate(dataloader):
            # Store original images and paths for later visualization if save_visuals is enabled
            if save_visuals:
                original_images.append(sample.cpu())  # Store the tensor
                original_paths.append(path)  # Store the path string

            gt_list_sp.extend(t2np(label))
            gt_list_px.extend(t2np(gt))
            weights_cnt += 1

            img = sample.to(device)
            t_tf, de_features = model(img)

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                output_list[l].append(output)
        fps = len(dataloader.dataset) / (time.time() - start_time)
        print("fps:", fps, len(dataloader.dataset))

        # postprocess
        anomaly_score, anomaly_map = weighted_decision_mechanism(
            weights_cnt, output_list, c.alpha, c.beta
        )

        gt_label = np.asarray(gt_list_sp, dtype=np.bool_)
        gt_mask = np.squeeze(np.asarray(gt_list_px, dtype=np.bool_), axis=1)

        anomaly_map = (
            F.interpolate(
                torch.from_numpy(anomaly_map).unsqueeze(1),
                size=gt_mask.shape[1:],  # gt_mask의 (H, W) 크기를 목표로 설정
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .numpy()
        )

        auroc_px = round(
            roc_auc_score(gt_mask.flatten(), anomaly_map.flatten()) * 100, 1
        )
        auroc_sp = round(roc_auc_score(gt_label, anomaly_score) * 100, 1)

        ap = round(average_precision_score(gt_label, anomaly_score) * 100, 1)
        pro = round(eval_seg_pro(gt_mask, anomaly_map), 1)

    if save_visuals:
        abnormal_indices = np.where(gt_label == 1)[0]

        base_visual_save_dir = os.path.join(c.save_dir, "visuals", "DAC", c._class_)

        print(
            f"Saving visuals for {len(abnormal_indices)} abnormal samples to {base_visual_save_dir}..."
        )

        for i in abnormal_indices:
            map_np = anomaly_map[i]
            map_ts = torch.tensor(map_np)
            full_image_path = original_paths[i][0]
            print(type(full_image_path))
            print(full_image_path)

            path_parts = full_image_path.split(os.sep)
            anomaly_type = path_parts[-2]
            image_filename = os.path.basename(full_image_path)
            root, ext = os.path.splitext(image_filename)
            image_filename = f"{root}_map{ext}"

            final_save_dir = os.path.join(base_visual_save_dir, anomaly_type)
            final_save_path = os.path.join(final_save_dir, image_filename)

            save_anomaly_visualization(
                anomaly_map=map_ts,
                save_path=final_save_path,
            )

    return auroc_px, auroc_sp, pro, ap


def eval_seg_pro(gt_mask, anomaly_score_map, max_step=800):
    expect_fpr = 0.3  # default 30%

    max_th = anomaly_score_map.max()
    min_th = anomaly_score_map.min()
    delta = (max_th - min_th) / max_step
    threds = np.arange(min_th, max_th, delta).tolist()

    pool = Pool(8)
    ret = pool.map(partial(single_process, anomaly_score_map, gt_mask), threds)
    pool.close()
    pros_mean = []
    fprs = []
    for pro_mean, fpr in ret:
        pros_mean.append(pro_mean)
        fprs.append(fpr)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)
    # expect_fpr = sum(fprs) / len(fprs)
    idx = (
        fprs < expect_fpr
    )  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    loc_pro_auc = auc(fprs_selected, pros_mean_selected) * 100

    return loc_pro_auc


def single_process(anomaly_score_map, gt_mask, thred):
    binary_score_maps = np.zeros_like(anomaly_score_map, dtype=np.bool_)
    binary_score_maps[anomaly_score_map <= thred] = 0
    binary_score_maps[anomaly_score_map > thred] = 1
    pro = []
    for binary_map, mask in zip(binary_score_maps, gt_mask):  # for i th image
        for region in regionprops(label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pro.append(tp_pixels / region.area)

    pros_mean = np.array(pro).mean()
    inverse_masks = 1 - gt_mask
    fpr = np.logical_and(inverse_masks, binary_score_maps).sum() / inverse_masks.sum()
    return pros_mean, fpr


def evaluation_batch(
    c, model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0
):
    model.train_or_eval(type="eval")
    gt_list_sp = []
    output_list = [list() for i in range(6)]
    weights_cnt = 0

    with torch.no_grad():
        for img, gt, label, cls in dataloader:
            img = img.to(device)
            gt_list_sp.extend(t2np(label))
            t_tf, de_features = model(img)
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                # print(output, output.shape)
                output_list[l].append(output)

        anomaly_score, _ = weighted_decision_mechanism(
            weights_cnt, output_list, c.alpha, c.beta
        )

        # anomaly_score = gaussian_filter(anomaly_score, sigma=4)
        gt_list_sp = np.asarray(gt_list_sp, dtype=np.bool_)
        # pr_list_sp.extend(sp_score)

        auroc_sp = round(roc_auc_score(gt_list_sp, anomaly_score), 4)
        ap_sp = round(average_precision_score(gt_list_sp, anomaly_score), 4)
        f1_sp = f1_score_max(gt_list_sp, anomaly_score)

    return auroc_sp, ap_sp, f1_sp


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def extract_numbers(file_name):
    numbers = re.findall(r"(\d+)", file_name)
    return tuple(map(int, numbers))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return 1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr + 1e-8))


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    max_ele = np.max(psnr_list)
    min_ele = np.min(psnr_list)
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], max_ele, min_ele))

    return anomaly_score_list
