from PIL import Image
import os
import torch
import glob
import numpy as np
import cv2

from torchvision import transforms as T

import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="importlib._bootstrap"
)


def loading_dataset(c, dataset_name):
    DATASET_CLASSES = {
        "DAC": DACDataset,
    }

    DatasetClass = DATASET_CLASSES.get(dataset_name)
    if not DatasetClass:
        print("Dataset does not exist")
        return None, None

    train_data = DatasetClass(c, is_train=True)
    test_data = DatasetClass(c, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=c.batch_size, shuffle=True, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, pin_memory=True
    )

    return train_dataloader, test_dataloader


class DarkenGlare:
    def __init__(
        self,
        lower=(200, 200, 200),
        upper=(255, 255, 255),
        strength=0.3,
        dilate=3,
        feather=7,
        use_hsv_thr=False,
        v_pct=99.0,
        v_abs=230,
        s_max=80,
    ):
        self.lower, self.upper = lower, upper
        self.strength, self.dilate, self.feather = strength, dilate, feather
        self.use_hsv_thr, self.v_pct, self.v_abs, self.s_max = (
            use_hsv_thr,
            v_pct,
            v_abs,
            s_max,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode == "RGB":
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            alpha = None
        elif img.mode == "RGBA":
            rgba = np.asarray(img)
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            alpha = rgba[..., 3]
        elif img.mode == "L":
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_GRAY2BGR)
            alpha = None
        else:
            bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            alpha = None

        if self.use_hsv_thr:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            V, S = hsv[..., 2], hsv[..., 1]
            thr = max(int(np.percentile(V, self.v_pct)), self.v_abs)
            mask = ((V >= thr) & (S <= self.s_max)).astype(np.uint8) * 255
        else:
            mask = cv2.inRange(bgr, self.lower, self.upper)

        if self.dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate, self.dilate))
            mask = cv2.dilate(mask, k, 1)
        m = mask.astype(np.float32) / 255.0
        if self.feather > 0:
            ksz = self.feather | 1
            m = cv2.GaussianBlur(m, (ksz, ksz), 0)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        Vf = np.clip(V.astype(np.float32) * (1.0 - self.strength * m), 0, 255).astype(
            np.uint8
        )
        out_bgr = cv2.cvtColor(cv2.merge([H, S, Vf]), cv2.COLOR_HSV2BGR)

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        if alpha is not None:
            out = Image.fromarray(np.dstack([out_rgb, alpha]), mode="RGBA")
        else:
            out = Image.fromarray(out_rgb, mode="RGB")
        return out


class DACDataset(torch.utils.data.Dataset):
    def __init__(self, c, is_train=True, dataset="DAC_noaug"):
        self.image_size = (c.image_size, c.image_size)

        # self.pre_transform = DarkenGlare(use_hsv_thr=True)
        self.pre_transform = None

        self.transform_img = T.Compose(
            [
                T.CenterCrop((2592, 416)),
                T.Resize(self.image_size, interpolation=T.InterpolationMode.LANCZOS),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_mask = T.Compose(
            [
                T.CenterCrop((2592, 416)),
                T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

        self.dataset_path = "../data/" + dataset
        self.phase = "train" if is_train else "test"
        self.class_name = c._class_
        self.img_dir = os.path.join(self.dataset_path, self.class_name, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        self.x, self.y, self.mask = self.load_dataset()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_path, y, mask_path = self.x[idx], self.y[idx], self.mask[idx]

        img = Image.open(x_path).convert("RGB")
        if mask_path is None:
            mask = Image.new("L", img.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")

        if self.pre_transform:
            img = self.pre_transform(img)

        x_tensor = self.transform_img(img)
        mask_tensor = self.transform_mask(mask)

        return x_tensor, y, mask_tensor, x_path

    def load_dataset(self):
        img_tot_paths = list()
        gt_tot_paths = list()
        tot_labels = list()

        defect_types = os.listdir(self.img_dir)

        for defect_type in defect_types:
            if defect_type == "good":
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([None] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                gt_paths = glob.glob(os.path.join(self.gt_dir, defect_type) + "/*")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))

        assert len(img_tot_paths) == len(
            tot_labels
        ), "Something wrong with test and ground truth pair!"

        return img_tot_paths, tot_labels, gt_tot_paths
