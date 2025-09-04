import copy

import torch
import numpy as np
import time
import os

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from datasets import loading_dataset
from utils import save_weights, to_device
from UniNet_lib.model import UniNet, EarlyStopping
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from eval import evaluation_indusAD


def train(c):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_name = "DAC"
    ckpt_path = os.path.join("./ckpts", dataset_name)

    # ---------------------------------loading dataset-----------------------------------------------
    train_dataloader, test_dataloader = loading_dataset(c, dataset_name)

    # ---------------------------------loading model-------------------------------------------------
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None
    Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device(
        [Source_teacher, bn, student, DFS], device
    )
    Target_teacher = copy.deepcopy(Source_teacher)

    params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=c.lr_s, betas=(0.9, 0.999), weight_decay=1e-5
    )
    optimizer1 = torch.optim.AdamW(
        list(Target_teacher.parameters()),
        lr=1e-4 if c._class_ == "transistor" else c.lr_t,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=DFS)

    auroc_sp, auroc_px, aupro_px, ap, max_IRoc, max_PRoc, max_PPro, max_AP = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    early_stopping = EarlyStopping(patience=c.patience, verbose=False)

    # ---------------------------------------------training-----------------------------------------------
    for epoch in range(c.epochs):
        print(f"training epoch: {epoch + 1}")
        model.train_or_eval(type="train")
        loss_list = []
        start_time = time.time()
        for sample in train_dataloader:
            img = sample[0].to(device)
            loss = model(img, stop_gradient=False)
            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            optimizer1.step()
            loss_list.append(loss.item())

            # ------------------------------------eval industrial and video-------------------------------------

        epoch_duration = time.time() - start_time
        print(
            "epoch [{}/{}], loss:{:.4f}, time:{:.2f}s".format(
                epoch + 1, c.epochs, np.mean(loss_list), epoch_duration
            )
        )

        modules_list = [model.t.t_t, model.bn.bn, model.s.s1, DFS]
        if (epoch + 1) % 10 == 0:

            # evaluation
            auroc_px, auroc_sp, aupro_px, ap = evaluation_indusAD(
                c, model, test_dataloader, device
            )
            print(
                "Sample Auroc: {:.1f}, AP: {:.1f}, Pixel Auroc: {:.1f}, Pixel Aupro: {:.1f}".format(
                    auroc_sp, ap, auroc_px, aupro_px
                )
            )

            # Track all max values and save weights if a metric improves
            if max_IRoc < auroc_sp:
                max_IRoc = auroc_sp
                print("saved BEST_I_ROC")
                (
                    save_weights(modules_list, ckpt_path, "BEST_I_ROC")
                    if c.is_saved
                    else None
                )
            if max_PPro < aupro_px:
                max_PPro = aupro_px
                print("saved BEST_P_PRO")
                (
                    save_weights(modules_list, ckpt_path, "BEST_P_PRO")
                    if c.is_saved
                    else None
                )
            if max_AP < ap:
                max_AP = ap
                print("saved BEST_AP")
                (
                    save_weights(modules_list, ckpt_path, "BEST_AP")
                    if c.is_saved
                    else None
                )
            if max_PRoc < auroc_px:
                max_PRoc = auroc_px

            print(
                f"MAX I_ROC: {max_IRoc:.1f}, MAX AP: {max_AP:.1f}, MAX P_ROC: {max_PRoc:.1f}, MAX P_PRO: {max_PPro:.1f}"
            )
            early_stopping(aupro_px)
            if early_stopping.early_stop:
                print("Early stopping")
                break
