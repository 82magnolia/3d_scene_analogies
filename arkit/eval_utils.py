import torch
from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.loss import chamfer_distance
from typing import List, Union, Dict
import pickle


def map_accuracy(transform, eval_points: torch.Tensor, gt_points: torch.Tensor, thres_list: List[float], consider_reflection_symmetry: bool = True):
    transform_points = transform(eval_points)

    # Align transformed points with ground-truth points only considering reflection
    centered_transform_points = transform_points - transform_points.mean(dim=0, keepdims=True)
    centered_gt_points = gt_points - gt_points.mean(dim=0, keepdims=True)
    align_ortho, _, _ = corresponding_points_alignment(
        centered_transform_points[None, ...],
        centered_gt_points[None, ...],
        estimate_scale=False,
        allow_reflection=True
    )
    align_ortho = align_ortho[0]
    loss = (transform_points - gt_points).norm(dim=-1)  # Cost not considering symmetry

    if consider_reflection_symmetry and torch.det(align_ortho).item() < 0.:  # Reflection matrix for considering symmetry
        align_transform_points = centered_transform_points @ align_ortho
        symmetry_thres = 0.4  # Hard-coded threshold for determining symmetry (NOTE: We only consider reflection symmetry and ignore rotation symmetry)
        symmetry_dist = chamfer_distance(align_transform_points[None, :], centered_transform_points[None, :])[0].sqrt().item()
        if symmetry_dist < symmetry_thres:  # Object group is symmetric
            align_loss = (align_transform_points - centered_gt_points).norm(dim=-1)
            loss = torch.minimum(align_loss, loss)

    # Returns list of accuracies
    acc_list = []
    for thres in thres_list:
        acc = (loss < thres).sum() / loss.shape[0]
        acc_list.append(acc.item())

    return acc_list


# Helper function used for evaluating two matching point sets only considering symmetry
def symmetry_aware_point_accuracy(eval_points: torch.Tensor, gt_points: torch.Tensor, thres_list: List[float]):
    # Align transformed points with ground-truth points only considering reflection
    centered_eval_points = eval_points - eval_points.mean(dim=0, keepdims=True)
    centered_gt_points = gt_points - gt_points.mean(dim=0, keepdims=True)
    align_ortho, _, _ = corresponding_points_alignment(
        centered_eval_points[None, ...],
        centered_gt_points[None, ...],
        estimate_scale=False,
        allow_reflection=True
    )
    align_ortho = align_ortho[0]
    loss = (eval_points - gt_points).norm(dim=-1)  # Cost not considering symmetry

    if torch.det(align_ortho).item() < 0.:  # Reflection matrix for considering symmetry
        align_eval_points = centered_eval_points @ align_ortho
        symmetry_thres = 0.4  # Hard-coded threshold for determining symmetry (NOTE: We only consider reflection symmetry and ignore rotation symmetry)
        symmetry_dist = chamfer_distance(align_eval_points[None, :], centered_eval_points[None, :])[0].sqrt().item()
        if symmetry_dist < symmetry_thres:  # Object group is symmetric
            align_loss = (align_eval_points - centered_gt_points).norm(dim=-1)
            loss = torch.minimum(align_loss, loss)

    # Returns list of accuracies
    acc_list = []
    for thres in thres_list:
        acc = (loss < thres).sum() / loss.shape[0]
        acc_list.append(acc.item())

    return acc_list


def chamfer_accuracy(transform, eval_points: torch.Tensor, gt_points: torch.Tensor, thres_list: List[float], eval_inst_labels: torch.Tensor = None, gt_inst_labels: torch.Tensor = None,
    eval_sem_labels: torch.Tensor = None, gt_sem_labels: torch.Tensor = None):
    # NOTE: We assume gt_points are supersets of transformed eval_points
    transform_points = transform(eval_points)

    if eval_inst_labels is None:  # If no instance labels are provided, compute one-sided Chamfer loss (eval points to nearest ground-truth points)
        cmf_result = chamfer_distance(gt_points[None, :], transform_points[None, :], batch_reduction=None, point_reduction=None)
        loss = cmf_result[0][1].sqrt().mean().item()  # NN-distance from transform points to ground-truth points

        # Return list of accuracies
        acc_list = []
        for thres in thres_list:
            acc = float(loss < thres)
            acc_list.append(acc)
        return acc_list
    else:  # If instance labels are provided, compute two-sided Chamfer loss (eval points to nearest ground-truth instances)
        # Compute centroid locations of eval and ground-truth points
        eval_centroids = []
        eval_unq_insts = eval_inst_labels.unique()
        for unq_inst in eval_unq_insts:
            eval_centroids.append(transform_points[eval_inst_labels == unq_inst].mean(0))
        eval_centroids = torch.stack(eval_centroids, dim=0)

        gt_centroids = []
        gt_unq_insts = gt_inst_labels.unique()
        for unq_inst in gt_unq_insts:
            gt_centroids.append(gt_points[gt_inst_labels == unq_inst].mean(0))
        gt_centroids = torch.stack(gt_centroids, dim=0)

        match_unq_insts = gt_unq_insts[(eval_centroids[:, None, :] - gt_centroids[None, :, :]).norm(dim=-1).argmin(dim=-1)]
        loss = 0.
        for eval_unq_inst, match_unq_inst in zip(eval_unq_insts, match_unq_insts):
            cmf_result = chamfer_distance(gt_points[None, gt_inst_labels == match_unq_inst], transform_points[None, eval_inst_labels == eval_unq_inst], batch_reduction=None, point_reduction=None)
            gt2eval_loss = cmf_result[0][0].sqrt().mean().item()  # NN-distance from transform points to ground-truth points
            eval2gt_loss = cmf_result[0][1].sqrt().mean().item()  # NN-distance from transform points to ground-truth points
            curr_loss = (gt2eval_loss + eval2gt_loss) / 2.

            if eval_sem_labels is not None and gt_sem_labels is not None:  # Additionally consider semantic labels during evaluation
                if gt_sem_labels[gt_inst_labels == match_unq_inst][0] != eval_sem_labels[eval_inst_labels == eval_unq_inst][0]:
                    curr_loss = 1000.  # Arbitrarily large value for having the sample to be considered invalid
            loss += curr_loss
        loss = loss / len(eval_unq_insts)
        # Return list of accuracies
        acc_list = []
        for thres in thres_list:
            acc = float(loss < thres)
            acc_list.append(acc)
        return acc_list


class MetricLogger:
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.metric_avg = {"avg_" + m: None for m in metrics}
        self.metric_full = {m: [] for m in metrics}
        self.metric_latest = {"curr_" + m: None for m in metrics}
        self.metric_num_data_points = {m: [] for m in metrics}

    def update_values(self, new_metric: Dict[str, Union[float, int]], num_data_points: Dict[str, Union[float, int]]):
        for m in new_metric.keys():
            self.metric_full[m].extend([new_metric[m]] * num_data_points[m])
            self.metric_num_data_points[m].append(num_data_points[m])
            self.metric_avg["avg_" + m] = sum(self.metric_full[m]) / sum(self.metric_num_data_points[m])
            self.metric_latest["curr_" + m] = new_metric[m]

    def save_to_path(self, path: str):
        save_dict = {
            'avg': self.metric_avg,
            'full': self.metric_full
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
