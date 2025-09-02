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
        "MVTecAD": MVTecDataset,
        "MTD": MtdDataset,
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
        # PIL -> ndarray(BGR)
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

        # 마스크
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

        # HSV V만 감산
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        Vf = np.clip(V.astype(np.float32) * (1.0 - self.strength * m), 0, 255).astype(
            np.uint8
        )
        out_bgr = cv2.cvtColor(cv2.merge([H, S, Vf]), cv2.COLOR_HSV2BGR)

        # ndarray(BGR) -> PIL
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        if alpha is not None:
            out = Image.fromarray(np.dstack([out_rgb, alpha]), mode="RGBA")
        else:
            out = Image.fromarray(out_rgb, mode="RGB")
        return out


class UnifiedTransform:
    """
    Applies aspect-ratio-preserving resize (letterboxing) to both image and mask,
    and generates a padding exclusion mask.
    """

    def __init__(self, image_size, fill_color=0, pre_transform=None):
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size  # (H, W)
        self.fill_color = fill_color
        self.to_tensor = T.ToTensor()
        self.pre_transform = pre_transform

    def __call__(self, img: Image.Image, mask: Image.Image):
        # 0. Apply pre-processing transforms like DarkenGlare if provided
        if self.pre_transform:
            img = self.pre_transform(img)

        # 1. Calculate resize geometry from the original or pre-processed image
        w, h = img.size
        target_h, target_w = self.image_size
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2

        # 2. Transform the image
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_img = Image.new(img.mode, (target_w, target_h), self.fill_color)
        padded_img.paste(resized_img, (paste_x, paste_y))
        img_tensor = self.to_tensor(padded_img)

        # 3. Transform the mask using the same geometry
        resized_mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        padded_mask_img = Image.new(mask.mode, (target_w, target_h), 0)
        padded_mask_img.paste(resized_mask, (paste_x, paste_y))
        mask_tensor = self.to_tensor(padded_mask_img)

        # 4. Create padding exclusion mask
        padding_mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        padding_mask[:, paste_y : paste_y + new_h, paste_x : paste_x + new_w] = 1.0

        return img_tensor, mask_tensor, padding_mask


class BaseADDataset(torch.utils.data.Dataset):
    """Base class for anomaly detection datasets to handle common transforms."""

    def __init__(self, c, is_train=True, dataset="MVTecAD"):
        self.is_train = is_train

        # Define pre-processing transforms to be applied before resizing
        pre_transform = DarkenGlare(use_hsv_thr=True)

        # Unified transform for image, mask, and padding mask generation
        # It now includes the pre-processing step.
        self.transform = UnifiedTransform(
            image_size=c.image_size, pre_transform=pre_transform
        )

        self.normalize = T.Compose(
            [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def __getitem__(self, idx):
        x_path, y, mask_path = self.x[idx], self.y[idx], self.mask[idx]
        # Use .convert('RGB') to robustly handle both color and grayscale images,
        x = Image.open(x_path).convert("RGB")

        if mask_path is None:
            mask = Image.new("L", x.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")

        x_tensor, mask_tensor, padding_mask = self.transform(x, mask)

        x_tensor = self.normalize(x_tensor)

        return x_tensor, y, mask_tensor, padding_mask, x_path

    def __len__(self):
        return len(self.x)


class MVTecDataset(BaseADDataset):
    def __init__(self, c, is_train=True, dataset="MVTecAD"):
        super().__init__(c, is_train)
        self.dataset_path = "../../../data/" + dataset
        self.class_name = c._class_
        phase = "train" if self.is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")
        # load dataset
        self.x, self.y, self.mask, _ = self.load_dataset()

    def load_dataset(self):

        img_tot_paths = list()
        gt_tot_paths = list()
        tot_labels = list()

        defect_types = os.listdir(self.img_dir)

        for defect_type in defect_types:
            # if self.is_vis and defect_type == "good":
            # continue
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


class MtdDataset(BaseADDataset):
    def __init__(self, c, is_train=True, dataset="MTD_exp"):
        super().__init__(c, is_train)
        self.dataset_path = "../../../data/" + dataset
        self.phase = "train" if is_train else "test"
        self.img_dir = os.path.join(self.dataset_path, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, "ground_truth")

        # load dataset
        self.x, self.y, self.mask = self.load_dataset()

    def load_dataset(self):
        img_paths = list()
        gt_paths = list()
        labels = list()

        defect_types = os.listdir(self.img_dir)

        for defect in defect_types:
            current_img_paths = glob.glob(os.path.join(self.img_dir, defect) + "/*")
            if not current_img_paths:
                continue  # Skip empty directories

            current_img_paths.sort()
            num_current_images = len(current_img_paths)
            img_paths.extend(current_img_paths)

            if defect == "good":
                gt_paths.extend([None] * num_current_images)
                labels.extend([0] * num_current_images)
            else:
                current_gt_paths = glob.glob(os.path.join(self.gt_dir, defect) + "/*")
                current_gt_paths.sort()
                # Add an assertion for robustness
                assert len(current_img_paths) == len(current_gt_paths), (
                    f"Mismatch in number of images and masks for defect type '{defect}' in phase '{self.phase}'. "
                    f"Found {len(current_img_paths)} images and {len(current_gt_paths)} masks."
                )
                gt_paths.extend(current_gt_paths)
                labels.extend([1] * num_current_images)

        assert len(img_paths) == len(
            labels
        ), f"Number of samples do not match for {self.phase}. Images: {len(img_paths)}, Labels: {len(labels)}"

        return img_paths, labels, gt_paths


class DACDataset(BaseADDataset):
    def __init__(self, c, is_train=True, dataset="DAC"):
        super().__init__(c, is_train)
        self.dataset_path = "../../../data/" + dataset
        self.phase = "train" if is_train else "test"
        self.class_name = c._class_
        self.img_dir = os.path.join(self.dataset_path, self.class_name, self.phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        # load dataset
        self.x, self.y, self.mask = self.load_dataset()

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


if __name__ == "__main__":
    dac_dataset = DACDataset()
    print(dac_dataset)
