import numpy as np
import os

from train_unsupervisedAD import train
from metadata import (
    unsupervised,
    mvtec_dict,
    industrial,
)
import argparse
from utils import setup_seed
from test import test


def parsing_args():
    parser = argparse.ArgumentParser(description="UniNet")

    parser.add_argument(
        "--domain",
        default="industrial",
        type=str,
        choices=["industrial"],
        help="choose experimental domain.",
    )
    parser.add_argument(
        "--dataset",
        default="MVTecAD",
        type=str,
        choices=["MVTecAD", "MTD"],
        help="choose experimental dataset.",
    )
    parser.add_argument(
        "--class_group",
        default="all",
        type=str,
        choices=["all", "texture", "object"],
        help="For MVTecAD, choose a class group to run.",
    )
    parser.add_argument(
        "--task",
        default="ad",
        type=str,
        choices=["ad", "as"],
        help="choose task between anomaly detection & segmentation.",
    )

    parser.add_argument("--epochs", default=100, type=int, help="epochs.")
    parser.add_argument("--batch_size", default=8, type=int, help="batch sizes.")
    parser.add_argument("--image_size", default=256, type=int, help="image size.")
    parser.add_argument("--center_crop", default=256, type=int, help="crop image size.")
    parser.add_argument(
        "--lr_s", default=5e-3, type=float, help="lr for student."
    )  # 5e-3
    parser.add_argument(
        "--lr_t", default=1e-6, type=float, help="lr for teacher."
    )  # 1e-6
    parser.add_argument(
        "--T", default=2, type=float, help="temperature for contrastive learning."
    )

    parser.add_argument(
        "--weighted_decision_mechanism",
        action="store_true",
        default=True,
        help="whether to employ weight-guided similarity to calculate anomaly map.",
    )
    parser.add_argument(
        "--default", default=0.3, type=float, help="the default value of weights."
    )
    parser.add_argument(
        "--alpha", default=0.01, type=float, help="hyperparameters for weights."
    )
    parser.add_argument(
        "--beta", default=0.00003, type=float, help="hyperparameters for weights."
    )

    parser.add_argument(
        "--is_saved",
        action="store_true",
        default=True,
        help="whether to save model weights.",
    )
    parser.add_argument("--save_dir", type=str, default="../../../data")
    parser.add_argument(
        "--load_ckpts",
        action="store_true",
        default=False,
        help="loading ckpts for testing",
    )
    parser.add_argument(
        "--save_visuals",
        action="store_true",
        default=False,
        help="Save anomaly map visualizations for abnormal test samples.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    setup_seed(1203)
    c = parsing_args()
    if not c.weighted_decision_mechanism:
        c.default = c.alpha = c.beta = c.gamma = "w/o"

    dataset_name = c.dataset
    dataset_class_group = c.class_group

    dataset = None
    if dataset_name in industrial:
        c.domain = "industrial"
        if dataset_name == "MVTecAD":
            if c.class_group == "all":
                dataset = mvtec_dict["texture"] + mvtec_dict["object"]
            else:
                dataset = mvtec_dict[c.class_group]
        elif dataset_name == "MTD":
            dataset = ["MTD"]

    else:
        raise KeyError(f"Dataset '{dataset_name}' can not be found.")

    from tabulate import tabulate

    results = {}
    table_ls = []

    # ---------------------------------------------------------------------------------------------------------
    # --------------------------------------unsupervised industrial AD-----------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    if dataset_name in industrial and dataset_name in unsupervised:
        # -----------------------------train-------------------------------------
        if not c.load_ckpts:
            for idx, i in enumerate(dataset):
                c._class_ = i

                args_dict = vars(c)
                args_info = f"class:{i}, "
                for key, value in args_dict.items():
                    if key in ["_class_"]:
                        continue
                    args_info += ", ".join([f"{key}:{value}, "])

                if idx == 0:
                    print(
                        f"training on {dataset_name} dataset for {dataset_class_group} classes"
                    )

                print(args_info)
                train(c)

            print("training over!")

            # -----------------------------test-------------------------------------
            # Initialize lists and headers based on task
        if c.task == "ad":
            headers = ["class", "I-AUROC", "AP"]
            metric_list_1 = []  # I-AUROC
            metric_list_2 = []  # AP
        else:  # c.task == 'as'
            headers = ["class", "P-AUROC", "PRO"]
            metric_list_1 = []  # P-AUROC
            metric_list_2 = []  # PRO

        for idx, i in enumerate(dataset):
            (
                print(
                    f"testing on {dataset_name} dataset for {dataset_class_group} classes"
                )
                if idx == 0
                else None
            )
            c._class_ = i
            print(f"testing class:{i}")

            # For AS, test with the model best at segmentation (P-PRO)
            # For AD, test with the model best at detection (I-ROC)
            if c.task == "as":
                auroc_sp, auroc_px, aupro_px, ap = test(
                    c, suffix="BEST_P_PRO", save_visuals=c.save_visuals
                )
            else:
                auroc_sp, auroc_px, aupro_px, ap = test(
                    c, suffix="BEST_I_ROC", save_visuals=c.save_visuals
                )

            print("")

            # Append results based on task
            if c.task == "ad":
                row = [
                    f"{i}",
                    f"{auroc_sp:.1f}",
                    f"{ap:.1f}",
                ]
                metric_list_1.append(auroc_sp)
                metric_list_2.append(ap)
            else:  # c.task == 'as'
                row = [
                    f"{i}",
                    f"{auroc_px:.1f}",
                    f"{aupro_px:.1f}",
                ]
                metric_list_1.append(auroc_px)
                metric_list_2.append(aupro_px)

            table_ls.append(row)
            results = tabulate(table_ls, headers=headers, tablefmt="pipe")

        # Append mean values
        table_ls.append(
            [
                "mean",
                f"{np.mean(metric_list_1):.2f}",
                f"{np.mean(metric_list_2):.2f}",
            ]
        )
        results = tabulate(table_ls, headers=headers, tablefmt="pipe")
        print(results)

        # settings
        param_grid = {
            "epochs": c.epochs,
            "batch_size": c.batch_size,
            "image_size": c.image_size,
            "center_crop": c.center_crop,
            "lr_s": c.lr_s,
            "lr_t": c.lr_t,
            "T": c.T,
            "alpha": c.alpha,
            "beta": c.beta,
        }

        # Save results to a file
        if c.task == "as":
            task_name = "segmentation"
        else:
            task_name = "detection"

        result_path = os.path.join("./saved_results", c.dataset, task_name)
        os.makedirs(result_path, exist_ok=True)
        result_file_path = os.path.join(result_path, f"{c.class_group}.txt")
        with open(result_file_path, "w") as f:
            f.write(
                f"Results for dataset: {c.dataset}, class group: {c.class_group}, task: {task_name}\n\n"
            )
            f.write(results)

            f.write(f"\n\nParameters:\n\n")
            f.write(str(param_grid))
