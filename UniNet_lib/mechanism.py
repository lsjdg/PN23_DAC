import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

"""Aggregates anomaly maps from multiple feature levels to compute a final anomaly map and score.

    This mechanism works in three main steps:
    1.  **Dynamic Weight Calculation:** For each image, it calculates a weight that determines the
        percentage of pixels to consider for the final score. This weight is derived from the
        most salient feature levels, identified by comparing the max anomaly values across levels.
    2.  **Anomaly Map Fusion:** It resizes all anomaly maps from all levels to a uniform size
        and sums them up to create a single, fused anomaly map for each image.
    3.  **Top-K Scoring:** The fused anomaly map is smoothed with a Gaussian filter. The final
        image-level anomaly score is the mean of the top-k pixel values from this smoothed map,
        where 'k' is determined by the dynamic weight calculated in step 1.

    Args:
        num (int): The number of samples in the batch.
        output_list (List[List[torch.Tensor]]): A list of lists of tensors.
            - The outer list corresponds to different feature levels.
            - The inner list contains anomaly map tensors for each sample in the batch.
            - e.g., output_list[level][sample_index] gives the anomaly map.
        alpha (float): A hyperparameter that scales the calculated weight for the top-k scoring.
        beta (float): A hyperparameter that provides a lower bound for the weight.
        out_size (int): The target resolution for the output anomaly maps.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]:
        - anomaly_score: A list of scalar anomaly scores for each sample, used for image-level detection.
        - anomaly_map: A numpy array of shape (num, out_size, out_size) representing the
          fused anomaly maps for the batch, used for pixel-level segmentation.
"""


def weighted_decision_mechanism(num, output_list, alpha, beta, out_size=256):

    total_weights_list = list()
    for i in range(num):
        low_similarity_list = list()
        for j in range(len(output_list)):
            low_similarity_list.append(torch.max(output_list[j][i]).cpu())
        probs = F.softmax(torch.tensor(low_similarity_list), 0)
        weight_list = (
            list()
        )  # set P consists of L high probability values, where L ranges from n-1 to n+1
        for idx, prob in enumerate(probs):
            (
                weight_list.append(low_similarity_list[idx].numpy())
                if prob > torch.mean(probs)
                else None
            )
        weight = np.max([np.mean(weight_list) * alpha, beta])
        total_weights_list.append(weight)

    assert (
        len(total_weights_list) == num
    ), "the number of weights dose not match that of samples!"

    am_lists = [list() for _ in output_list]
    for l, output in enumerate(output_list):
        output = torch.cat(output, dim=0)
        a_map = torch.unsqueeze(output, dim=1)  # B*1*h*w
        am_lists[l] = F.interpolate(
            a_map, size=out_size, mode="bilinear", align_corners=True
        )[
            :, 0, :, :
        ]  # B*256*256

    anomaly_map = sum(am_lists)

    anomaly_score_exp = anomaly_map
    batch = anomaly_score_exp.shape[0]
    anomaly_score = list()  # anomaly scores for all test samples
    for b in range(batch):
        top_k = int(out_size * out_size * total_weights_list[b])
        assert top_k >= 1 / (
            out_size * out_size
        ), "weight can not be smaller than 1 / (H * W)!"

        single_anomaly_score_exp = anomaly_score_exp[b]
        single_anomaly_score_exp = torch.tensor(
            gaussian_filter(single_anomaly_score_exp.detach().cpu().numpy(), sigma=4)
        )
        assert (
            single_anomaly_score_exp.reshape(1, -1).shape[-1] == out_size * out_size
        ), "something wrong with the last dimension of reshaped map!"

        single_map = single_anomaly_score_exp.reshape(1, -1)
        single_anomaly_score = np.mean(
            single_map.topk(top_k, dim=-1)[0].detach().cpu().numpy(), axis=1
        )
        anomaly_score.append(single_anomaly_score)

    return anomaly_score, anomaly_map.detach().cpu().numpy()
