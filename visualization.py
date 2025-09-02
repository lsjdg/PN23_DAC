import torch
import numpy as np
import cv2
import os


def save_anomaly_visualization(
    anomaly_map: torch.Tensor,
    save_path: str = None,
    colormap: int = cv2.COLORMAP_JET,
    show: bool = False,
):
    am = anomaly_map.detach().cpu().float()

    # (3,H,W) → (H,W,3)
    if am.ndim == 3 and am.shape[0] == 3:
        am = am.permute(1, 2, 0).numpy()
    elif am.ndim == 3 and am.shape[0] == 1:
        am = am.squeeze(0).numpy()
    else:
        am = am.numpy()

    # 정규화
    amin, amax = am.min(), am.max()
    if amax - amin < 1e-8:
        am01 = np.zeros_like(am, dtype=np.float32)
    else:
        am01 = (am - amin) / (amax - amin)

    if am01.ndim == 2:  # grayscale
        am_u8 = (am01 * 255).astype(np.uint8)
        am_color = cv2.applyColorMap(am_u8, colormap)
    else:  # 이미 3채널
        am_color = (am01 * 255).astype(np.uint8)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, am_color)

    if show:
        import matplotlib.pyplot as plt

        plt.imshow(cv2.cvtColor(am_color, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return am_color
