import torch
import torch.nn as nn
from typing import Dict, List
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from threed_front.rbf_interpolator import RBFInterpolator
from threed_front.utils import idw_nn_interpolation, box_align
import numpy as np


class PointMatcher:
    def __init__(self, cfg, device: torch.device, local_feature_field: nn.Module):
        self.cfg = cfg
        self.device = device
        self.local_feature_field = local_feature_field

        if self.cfg.local_cost_type == 'l2':
            self.cost_func = lambda x, y: (x - y).norm(dim=-1).mean()
        elif self.cfg.local_cost_type == 'l1':
            self.cost_func = lambda x, y: (x - y).abs().mean()
        else:
            raise NotImplementedError

    def find_transforms(self, global_transform_list: List[Dict[str, torch.Tensor]], global_inst_match_list: List[Dict[str, torch.Tensor]],
            tgt_query_pcd: Pointclouds, ref_query_pcd: Pointclouds, tgt_scene_pcd: Pointclouds, ref_scene_pcd: Pointclouds):
        # NOTE: We aim to find a transform that maps tgt (pair_pos) to ref (pos)
        num_batch_scenes = len(global_transform_list)
        local_transform_list = [[] for _ in range(num_batch_scenes)]
        local_inst_match_list = [[] for _ in range(num_batch_scenes)]
        for scene_idx in range(num_batch_scenes):
            # For each scene, list up global transformations and fit local matches
            cost_list = []
            warp_list = []  # List of warps used for suppressing duplicate local transforms
            for g_idx, global_transform in enumerate(tqdm(global_transform_list[scene_idx], desc=f"Local Matching (Scene {scene_idx + 1}, {len(global_transform_list[scene_idx])} Transforms)")):
                # Extract tgt (pair_pos) features
                with torch.no_grad():
                    tgt_query_points = tgt_query_pcd.points_packed()
                    ref_query_points = ref_query_pcd.points_packed()

                    tgt_feats = self.local_feature_field(tgt_query_pcd, tgt_scene_pcd)[0]  # (N_query, D_feat)
                    ref_feats = self.local_feature_field(ref_query_pcd, ref_scene_pcd)[0]  # (N_query, D_feat)

                    tgt_inst_labels = tgt_query_pcd.features_packed()[:, 0]
                    ref_inst_labels = ref_query_pcd.features_packed()[:, 0]
                    tgt_unq_inst = tgt_inst_labels.unique()

                # Generate interpolator from initial affine transform
                warp_query = tgt_query_points @ global_transform['A'].T + global_transform['T']

                # Align scale and centroids of query points to their nearest neighbors prior to local alignment
                if not getattr(self.cfg, "local_skip_box_align", False):
                    for inst_idx, tgt_inst_lab in enumerate(tgt_unq_inst):
                        if global_inst_match_list[scene_idx][g_idx][inst_idx] != -1:  # Align only for valid instance matches
                            match_inst_lab = global_inst_match_list[scene_idx][g_idx][inst_idx]
                            warp_inst_query = warp_query[tgt_inst_labels == tgt_inst_lab]
                            match_inst_query = ref_query_points[ref_inst_labels == match_inst_lab]
                            warp_query[tgt_inst_labels == tgt_inst_lab] = box_align(warp_inst_query, match_inst_query)

                warp_query = warp_query.requires_grad_()

                optimizer = torch.optim.Adam([warp_query], self.cfg.local_lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.cfg.local_patience, factor=self.cfg.local_factor)
                for it in range(self.cfg.local_num_iter):
                    optimizer.zero_grad()
                    # Warp with instance matches
                    cost = 0.
                    for inst_idx, tgt_inst_lab in enumerate(tgt_unq_inst):
                        if global_inst_match_list[scene_idx][g_idx][inst_idx] != -1:  # Optimize only for valid instance matches
                            match_inst_lab = global_inst_match_list[scene_idx][g_idx][inst_idx]
                            warp_inst_query = warp_query[tgt_inst_labels == tgt_inst_lab]
                            match_inst_query = ref_query_points[ref_inst_labels == match_inst_lab]
                            match_feats = ref_feats[ref_inst_labels == match_inst_lab]

                            warp_inst_query_pcd = Pointclouds(warp_inst_query.reshape(1, -1, 3))
                            match_inst_query_pcd = Pointclouds(match_inst_query.reshape(1, -1, 3))
                            warp_inst_feats = idw_nn_interpolation(warp_inst_query, match_inst_query, match_feats, dist_pow=1., agg_k=5)
                            tgt_inst_feats = tgt_feats[tgt_inst_labels == tgt_inst_lab]

                            if self.cfg.local_dist_cost_weight > 0.:
                                dist_cost = chamfer_distance(warp_inst_query_pcd, match_inst_query_pcd, batch_reduction=None, point_reduction=None)
                                dist_cost = torch.sqrt(dist_cost[0][0]).mean()
                            else:
                                dist_cost = 0.

                            if self.cfg.local_feat_cost_weight > 0.:
                                feat_cost = self.cost_func(tgt_inst_feats, warp_inst_feats)
                            else:
                                feat_cost = 0.

                            inst_cost = self.cfg.local_feat_cost_weight * feat_cost + self.cfg.local_dist_cost_weight * dist_cost
                            cost += inst_cost

                    if isinstance(cost, torch.Tensor) and cost.requires_grad:
                        cost.backward()
                    if warp_query.grad is not None and torch.isnan(warp_query.grad).sum() > 0:  # NaN loss occurs if alignment matches point pairs near perfectly (in this case, terminate)
                        break

                    optimizer.step()
                    scheduler.step(cost)
                    torch.cuda.empty_cache()

                # Cache average cost after alignment
                if isinstance(cost, torch.Tensor):
                    cost_list.append(cost.item() / tgt_unq_inst.shape[0])
                else:
                    cost_list.append(cost / tgt_unq_inst.shape[0])
                rbf_interpolator = RBFInterpolator(tgt_query_points, warp_query.clone().detach(), smoothing=self.cfg.local_rbf_smoothing, device=self.device)
                local_transform_list[scene_idx].append(rbf_interpolator.cpu())
                local_inst_match_list[scene_idx].append(global_inst_match_list[scene_idx][g_idx].cpu())
                warp_list.append(Pointclouds(warp_query.clone().detach().reshape(1, -1, 3)))

            # Non-maximum suppression
            if len(local_transform_list[scene_idx]) != 0:
                if self.cfg.local_nms_thres > 0.:
                    cost_arr = np.array(cost_list)

                    # Sort by feature cost of transforms
                    order = np.argsort(cost_arr)

                    # Picked transforms
                    picked_transforms = []
                    picked_inst_match = []
                    picked_score = []

                    # Iterate over transforms until all initial estimates are explained
                    while order.size > 0:
                        # The index of smallest cost
                        index = order[0]

                        # Pick the bounding box with largest cost
                        picked_transforms.append(local_transform_list[scene_idx][index])
                        picked_inst_match.append(local_inst_match_list[scene_idx][index])
                        picked_score.append(cost_list[index])

                        # Compute IoU between current transformed bbox and remaining transformed bboxes
                        curr_warp_query = warp_list[index]
                        cmf_dist_list = [torch.sqrt(chamfer_distance(curr_warp_query, warp_list[idx])[0]).item() for idx in order]
                        cmf_dist = np.array(cmf_dist_list)

                        left = np.where(cmf_dist > self.cfg.local_nms_thres)
                        order = order[left]

                    local_transform_list[scene_idx] = picked_transforms
                    local_inst_match_list[scene_idx] = picked_inst_match
                    cost_list = picked_score

                # Choose top-K smallest transforms that are over validity threshold
                cost_arr = np.array(cost_list)
                topk_list = np.argsort(cost_arr)[:self.cfg.local_topk].tolist()
                local_transform_list[scene_idx] = [local_transform_list[scene_idx][l_idx] for l_idx in topk_list if cost_arr[l_idx] < self.cfg.local_valid_thres]
                local_inst_match_list[scene_idx] = [local_inst_match_list[scene_idx][l_idx] for l_idx in topk_list if cost_arr[l_idx] < self.cfg.local_valid_thres]

        return local_transform_list, local_inst_match_list
