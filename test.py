import copy
import os

import torch

from UniNet_lib.DFS import DomainRelated_Feature_Selection
from eval import evaluation_indusAD, evaluation_batch
from UniNet_lib.resnet import wide_resnet50_2
from utils import load_weights, to_device
from datasets import loading_dataset
from metadata import unsupervised


def test(
    c, stu_type="un_cls", suffix="BEST_P_PRO", save_visuals=False
):  # Added save_visuals parameter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_name = c.dataset
    ckpt_path = None
    if c._class_ in [dataset_name]:
        ckpt_path = os.path.join("./ckpts", dataset_name)
    else:
        if c.setting == "oc":
            ckpt_path = os.path.join("./ckpts", dataset_name, f"{c._class_}")
        elif c.setting == "mc":
            ckpt_path = os.path.join("./ckpts", f"{dataset_name}", "multiclass")
        else:
            pass

    # --------------------------------------loading dataset------------------------------------------
    dataset_info = loading_dataset(c, dataset_name)
    test_dataloader = dataset_info[1]

    # ---------------------------------------loading model-------------------------------------------
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    # use first three layers
    Source_teacher.layer4 = None
    Source_teacher.fc = None

    # loading different student models
    student = None
    if stu_type == "un_cls":
        from UniNet_lib.de_resnet import de_wide_resnet50_2

        student = de_wide_resnet50_2(pretrained=False)

    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device(
        [Source_teacher, bn, student, DFS], device
    )
    Target_teacher = copy.deepcopy(Source_teacher)

    new_state = load_weights([Target_teacher, bn, student, DFS], ckpt_path, suffix)
    Target_teacher = new_state["tt"]
    bn = new_state["bn"]
    student = new_state["st"]
    DFS = new_state["dfs"]

    model = None
    if stu_type == "un_cls":
        from UniNet_lib.model import UniNet

        model = UniNet(
            c, Source_teacher.cuda().eval(), Target_teacher, bn, student, DFS
        )
        print("using UniNet model")

    if c.domain == "industrial":
        if c.setting == "oc":
            if dataset_name in unsupervised:
                auroc_px, auroc_sp, pro, ap = evaluation_indusAD(  # Pass save_visuals
                    c, model, test_dataloader, device, save_visuals=save_visuals
                )

                return auroc_sp, auroc_px, pro, ap

        else:  # multiclass
            auroc_sp_list, ap_sp_list, f1_list = [], [], []
            # test_dataloader: List
            for test_loader in test_dataloader:
                auroc_sp, ap_sp, f1 = evaluation_batch(c, model, test_loader, device)
                auroc_sp_list.append(auroc_sp * 100)
                ap_sp_list.append(ap_sp * 100)
                f1_list.append(f1 * 100)
            return auroc_sp_list, ap_sp_list, f1_list, dataset_info[-2]
