import torch.nn as nn
import torch
from threed_front.utils import generate_yaw_points, yaw2rot_mtx, rot_mtx2yaw, idw_nn_interpolation
import numpy as np
from pytorch3d.structures import Pointclouds
from typing import List, Dict
from tqdm import tqdm
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment


class AffineMatcher:
    def __init__(self, cfg, device: torch.device, global_feature_field: nn.Module):
        self.cfg = cfg
        self.device = device

        self.global_dist_valid_thres = self.cfg.global_dist_valid_thres
        self.global_feature_field = global_feature_field

        self.corner_idxs = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]).T  # Indices used for obtaining corners from query point boundaries

        # Initialize yaw values for affine alignment
        self.yaw_points = generate_yaw_points(self.cfg.global_num_rot)
        self.rot_matrices = yaw2rot_mtx(self.yaw_points, apply_xz_flip=True).float()

        if self.cfg.global_cost_type == 'l2':
            self.cost_func = lambda x, y: (x - y).norm(dim=-1).mean()
        elif self.cfg.global_cost_type == 'l1':
            self.cost_func = lambda x, y: (x - y).abs().mean()
        else:
            raise NotImplementedError

    def find_transforms(self, tgt_query_pcd: Pointclouds, ref_query_pcd: Pointclouds, tgt_scene_pcd: Pointclouds, ref_scene_pcd: Pointclouds, **kwargs):
        # NOTE 1: Affine matcher provides transformations DELTA such that tgt_query_pcd + DELTA matches ref_query_pcd
        # NOTE 2: Affine matcher assumes tgt_query_pcd comes from object surface points
        # NOTE 3: Affine matcher assumes ref_query_pcd comes from object surface points
        # NOTE 4: The matcher returns for each batch a list of transformations to match tgt_query_pcd to ref_query_pcd
        # NOTE 5: The equation for mapping volume in tgt to ref is as follows, X' = X @ A.T + T
        # NOTE 6: tgt corresponds to 'pair_pos' and ref corresponds to 'pos'
        # NOTE 7: init_tgt_query_pcd is set as mean of tgt_query_pcd

        # Flag for ignoring semantic labels
        if 'obj_subclasses' in kwargs:
            validation_modality = 'language_model'
            sentence_mapper = np.load(self.global_feature_field.cfg.sentence_emb_path)
            sentence_valid_thres = 3.0
        elif 'obj_id' in kwargs:
            if self.global_feature_field.cfg.semantics_emb_type == "clip":
                validation_modality = 'clip'
                clip_mapper = np.load(self.global_feature_field.cfg.clip_emb_path)
                clip_valid_thres = 12.0
            elif self.global_feature_field.cfg.semantics_emb_type == "dino":
                validation_modality = 'dino'
                dino_mapper = np.load(self.global_feature_field.cfg.dino_emb_path)
                dino_valid_thres = 12.0
            else:
                validation_modality = 'caption_model'
                caption_mapper = np.load(self.global_feature_field.cfg.caption_emb_path)
                caption_valid_thres = 8.0
        else:
            if getattr(self.global_feature_field, "semantics_emb_size", 0) != 0:
                validation_modality = 'semantics'
            else:  # Only distance embeddings are used
                validation_modality = 'distance'

        # Extract features
        with torch.no_grad():
            # Target scene features for refinement
            pair_pos_args = {key: kwargs[key]['pair_pos'] for key in kwargs.keys()}
            pos_args = {key: kwargs[key]['pos'] for key in kwargs.keys()}
            refine_tgt_feats = self.global_feature_field(tgt_query_pcd, tgt_scene_pcd, **pair_pos_args)  # (N_batch, N_pair_pos_query, D_feat)
            refine_tgt_query_points = tgt_query_pcd.points_padded()

            # Reference scene features for refinement
            refine_ref_feats = self.global_feature_field(ref_query_pcd, ref_scene_pcd, **pos_args)  # (N_batch, N_pos_query, D_feat)
            refine_ref_query_points = ref_query_pcd.points_padded()

        num_batches = len(tgt_query_pcd)

        # Bounds set for non-max suppression
        bounds_tgt = tgt_query_pcd.get_bounding_boxes().cpu()  # (N_batch, 3, 2)
        full_transform_list = []
        full_inst_match_list = []

        for b_idx in range(num_batches):
            # Extract per-object centroids and scales (NOTE: We assign height ratios for y-axis)
            tgt_inst_labels = tgt_query_pcd.features_list()[b_idx][:, 0]
            tgt_scene_feats = tgt_scene_pcd.features_list()[b_idx]
            tgt_unq_inst = tgt_inst_labels.unique()
            tgt_inst_centroids = torch.zeros(tgt_unq_inst.shape[0], 3)  # (N_obj_tgt, 3)
            tgt_inst_scales = torch.ones(tgt_unq_inst.shape[0], 3)  # (N_obj_tgt, 3) with maximum radius from xz bounding box (NOTE: Height ratios are not considered)
            tgt_inst_semantics = torch.zeros(tgt_unq_inst.shape[0], )  # (N_obj_tgt, )
            for lab_idx, lab in enumerate(tgt_unq_inst):
                tgt_inst_points = tgt_query_pcd.points_list()[b_idx][tgt_inst_labels == lab].cpu()
                tgt_inst_centroids[lab_idx, :] = (tgt_inst_points.max(dim=0).values + tgt_inst_points.min(dim=0).values) / 2.
                tgt_inst_scales[lab_idx, [0, 2]] = ((tgt_inst_points[:, [0, 2]].max(dim=0).values - tgt_inst_points[:, [0, 2]].min(dim=0).values) / 2.).norm(dim=-1)
                tgt_inst_semantics[lab_idx] = tgt_scene_feats[tgt_scene_feats[:, 0] == lab, 1][0]

            ref_inst_labels = ref_query_pcd.features_list()[b_idx][:, 0]
            ref_scene_feats = ref_scene_pcd.features_list()[b_idx]
            ref_unq_inst = ref_inst_labels.unique()
            ref_inst_centroids = torch.zeros(ref_unq_inst.shape[0], 3)  # (N_obj_ref, 3)
            ref_inst_scales = torch.ones(ref_unq_inst.shape[0], 3)  # (N_obj_ref, 3) with maximum radius from xz bounding box (NOTE: Height ratios are not considered)
            ref_inst_semantics = torch.zeros(ref_unq_inst.shape[0], )  # (N_obj_ref, )
            for lab_idx, lab in enumerate(ref_unq_inst):
                ref_inst_points = ref_query_pcd.points_list()[b_idx][ref_inst_labels == lab].cpu()
                ref_inst_centroids[lab_idx, :] = (ref_inst_points.max(dim=0).values + ref_inst_points.min(dim=0).values) / 2.
                ref_inst_scales[lab_idx, [0, 2]] = ((ref_inst_points[:, [0, 2]].max(dim=0).values - ref_inst_points[:, [0, 2]].min(dim=0).values) / 2.).norm(dim=-1)
                ref_inst_semantics[lab_idx] = ref_scene_feats[ref_scene_feats[:, 0] == lab, 1][0]

            # List initial translations from pair-wise object matches
            init_transform_list = []
            init_scores = []
            pbar = tqdm(total=tgt_unq_inst.shape[0] * ref_unq_inst.shape[0], desc=f"Initial Global Search (Scene {b_idx + 1} / {num_batches})")
            for tgt_inst_idx in range(tgt_unq_inst.shape[0]):
                for ref_inst_idx in range(ref_unq_inst.shape[0]):
                    init_tgt_match = tgt_inst_centroids[tgt_inst_idx: tgt_inst_idx + 1].clone().detach()
                    init_ref_match = ref_inst_centroids[ref_inst_idx: ref_inst_idx + 1].clone().detach()

                    # Align height values of object matches (NOTE: This will be refined during affine map optimization)
                    init_ref_match[:, 1] = init_tgt_match[:, 1]
                    init_scales = torch.diag(ref_inst_scales[ref_inst_idx] / tgt_inst_scales[tgt_inst_idx])

                    # Use validation modality to only consider valid object matches
                    if validation_modality == 'semantics':
                        if tgt_inst_semantics[tgt_inst_idx] != ref_inst_semantics[ref_inst_idx]:
                            rot = torch.eye(3).to(init_tgt_match.device)
                            trans = -(init_tgt_match @ init_scales.T) @ rot.T + init_ref_match
                            transform = {'R': rot, 'S': init_scales, 'A': rot @ init_scales, 'T': trans}
                            init_transform_list.append(transform)
                            init_scores.append(np.inf)
                            pbar.update()
                            continue
                    elif validation_modality == 'language_model':
                        tgt_inst_emb = sentence_mapper[kwargs["obj_subclasses"]['pair_pos'][b_idx][tgt_unq_inst[tgt_inst_idx].long().item()]]
                        ref_inst_emb = sentence_mapper[kwargs["obj_subclasses"]['pos'][b_idx][ref_unq_inst[ref_inst_idx].long().item()]]
                        if np.linalg.norm(tgt_inst_emb - ref_inst_emb, axis=-1) > sentence_valid_thres:
                            rot = torch.eye(3).to(init_tgt_match.device)
                            trans = -(init_tgt_match @ init_scales.T) @ rot.T + init_ref_match
                            transform = {'R': rot, 'S': init_scales, 'A': rot @ init_scales, 'T': trans}
                            init_transform_list.append(transform)
                            init_scores.append(np.inf)
                            pbar.update()
                            continue
                    elif validation_modality == 'clip':
                        tgt_inst_emb = clip_mapper[kwargs["obj_id"]['pair_pos'][b_idx][tgt_unq_inst[tgt_inst_idx].long().item()]]
                        ref_inst_emb = clip_mapper[kwargs["obj_id"]['pos'][b_idx][ref_unq_inst[ref_inst_idx].long().item()]]
                        if np.linalg.norm(tgt_inst_emb - ref_inst_emb, axis=-1) > clip_valid_thres:
                            rot = torch.eye(3).to(init_tgt_match.device)
                            trans = -(init_tgt_match @ init_scales.T) @ rot.T + init_ref_match
                            transform = {'R': rot, 'S': init_scales, 'A': rot @ init_scales, 'T': trans}
                            init_transform_list.append(transform)
                            init_scores.append(np.inf)
                            pbar.update()
                            continue
                    elif validation_modality == 'caption_model':
                        tgt_inst_emb = caption_mapper[kwargs["obj_id"]['pair_pos'][b_idx][tgt_unq_inst[tgt_inst_idx].long().item()]]
                        ref_inst_emb = caption_mapper[kwargs["obj_id"]['pos'][b_idx][ref_unq_inst[ref_inst_idx].long().item()]]
                        if np.linalg.norm(tgt_inst_emb - ref_inst_emb, axis=-1) > caption_valid_thres:
                            rot = torch.eye(3).to(init_tgt_match.device)
                            trans = -(init_tgt_match @ init_scales.T) @ rot.T + init_ref_match
                            transform = {'R': rot, 'S': init_scales, 'A': rot @ init_scales, 'T': trans}
                            init_transform_list.append(transform)
                            init_scores.append(np.inf)
                            pbar.update()
                            continue
                    elif validation_modality == 'dino':
                        tgt_inst_emb = dino_mapper[kwargs["obj_id"]['pair_pos'][b_idx][tgt_unq_inst[tgt_inst_idx].long().item()]]
                        ref_inst_emb = dino_mapper[kwargs["obj_id"]['pos'][b_idx][ref_unq_inst[ref_inst_idx].long().item()]]
                        if np.linalg.norm(tgt_inst_emb - ref_inst_emb, axis=-1) > dino_valid_thres:
                            rot = torch.eye(3).to(init_tgt_match.device)
                            trans = -(init_tgt_match @ init_scales.T) @ rot.T + init_ref_match
                            transform = {'R': rot, 'S': init_scales, 'A': rot @ init_scales, 'T': trans}
                            init_transform_list.append(transform)
                            init_scores.append(np.inf)
                            pbar.update()
                            continue

                    # NOTE: We assume a transformation of form X_ref = (X_tgt - init_tgt_match) @ scale.T @ R.T + init_ref_match = X_tgt @ R.T + t
                    for rot_idx, rot in enumerate(self.rot_matrices):
                        trans = -(init_tgt_match @ init_scales.T) @ rot.T + init_ref_match
                        # NOTE: 'R' and 'S' are only used as intermediate information for pose optimization
                        transform = {'R': rot, 'S': init_scales, 'A': rot @ init_scales, 'T': trans}
                        rot_score = self.evaluate_pose(transform, refine_tgt_query_points[b_idx], refine_tgt_feats[b_idx], refine_ref_query_points[b_idx], refine_ref_feats[b_idx])
                        init_transform_list.append(transform)
                        init_scores.append(rot_score)
                    pbar.update()
            pbar.close()
            init_scores = np.array(init_scores)
            topk_idx = np.argsort(init_scores)[:self.cfg.global_topk]
            transform_list = [init_transform_list[k_idx] for k_idx in topk_idx if init_scores[k_idx] < self.global_dist_valid_thres]
            cost_list = [init_scores[k_idx] for k_idx in topk_idx if init_scores[k_idx] < self.global_dist_valid_thres]

            if self.cfg.global_nms_thres > 0.:  # Optionally apply non-maximum suppresion of initial poses
                if self.cfg.global_nms_type == "bbox":
                    bbox_tgt = torch.take_along_dim(bounds_tgt[b_idx], self.corner_idxs, dim=-1).T  # (8, 3)
                    transform_list, cost_list = self.bbox_non_max_suppresion(bbox_tgt, transform_list, cost_list)
                elif self.cfg.global_nms_type == "transform":
                    transform_list, cost_list = self.transform_non_max_suppresion(transform_list, cost_list)
                else:
                    raise NotImplementedError("Other transforms not implemented")            

            if self.cfg.global_mapping_type == "affine":
                transform_list, cost_list = self.optimize_affine(b_idx, refine_tgt_query_points[b_idx], refine_tgt_feats[b_idx], refine_ref_query_points[b_idx], refine_ref_feats[b_idx], transform_list, True)
            elif self.cfg.global_mapping_type == "scale_rot":
                transform_list, cost_list = self.optimize_scale_rot(b_idx, refine_tgt_query_points[b_idx], refine_tgt_feats[b_idx], refine_ref_query_points[b_idx], refine_ref_feats[b_idx], transform_list, True)
            else:
                raise NotImplementedError("Other global mappings not supported")

            inst_match_list = []
            valid_transform_list = []

            # Create matrix specifying object-level match validity based on specified modalities
            if validation_modality == 'semantics':
                object_agreement = (tgt_inst_semantics[:, None] == ref_inst_semantics[None, :])
            elif validation_modality == 'language_model':
                tgt_inst_full_emb = np.stack([sentence_mapper[kwargs["obj_subclasses"]['pair_pos'][b_idx][inst.long().item()]] for inst in tgt_unq_inst], axis=0)
                ref_inst_full_emb = np.stack([sentence_mapper[kwargs["obj_subclasses"]['pos'][b_idx][inst.long().item()]] for inst in ref_unq_inst], axis=0)
                object_agreement = (np.linalg.norm(tgt_inst_full_emb[:, None] - ref_inst_full_emb[None, :], axis=-1) < sentence_valid_thres)
                object_agreement = torch.from_numpy(object_agreement)
            elif validation_modality == 'clip':
                tgt_inst_full_emb = np.stack([clip_mapper[kwargs["obj_id"]['pair_pos'][b_idx][inst.long().item()]] for inst in tgt_unq_inst], axis=0)
                ref_inst_full_emb = np.stack([clip_mapper[kwargs["obj_id"]['pos'][b_idx][inst.long().item()]] for inst in ref_unq_inst], axis=0)
                object_agreement = (np.linalg.norm(tgt_inst_full_emb[:, None] - ref_inst_full_emb[None, :], axis=-1) < clip_valid_thres)
                object_agreement = torch.from_numpy(object_agreement)
            elif validation_modality == 'caption_model':
                tgt_inst_full_emb = np.stack([caption_mapper[kwargs["obj_id"]['pair_pos'][b_idx][inst.long().item()]] for inst in tgt_unq_inst], axis=0)
                ref_inst_full_emb = np.stack([caption_mapper[kwargs["obj_id"]['pos'][b_idx][inst.long().item()]] for inst in ref_unq_inst], axis=0)
                object_agreement = (np.linalg.norm(tgt_inst_full_emb[:, None] - ref_inst_full_emb[None, :], axis=-1) < caption_valid_thres)
                object_agreement = torch.from_numpy(object_agreement)
            elif validation_modality == 'dino':
                tgt_inst_full_emb = np.stack([dino_mapper[kwargs["obj_id"]['pair_pos'][b_idx][inst.long().item()]] for inst in tgt_unq_inst], axis=0)
                ref_inst_full_emb = np.stack([dino_mapper[kwargs["obj_id"]['pos'][b_idx][inst.long().item()]] for inst in ref_unq_inst], axis=0)
                object_agreement = (np.linalg.norm(tgt_inst_full_emb[:, None] - ref_inst_full_emb[None, :], axis=-1) < dino_valid_thres)
                object_agreement = torch.from_numpy(object_agreement)
            else:
                object_agreement = torch.ones(tgt_inst_centroids.shape[0], ref_inst_centroids.shape[0], dtype=bool)

            inf_cost = 100.  # Large cost value assigned to semantically disagreeing object pairs
            for transform in transform_list:
                # Obtain instance-level matches for local refinement
                warp_inst_centroids = tgt_inst_centroids @ transform['A'].T + transform['T']
                centroid_dist = (warp_inst_centroids[:, None, :] - ref_inst_centroids[None, :, :]).norm(dim=-1)
                centroid_dist[~object_agreement] = inf_cost
                match_idx = torch.from_numpy(linear_sum_assignment(centroid_dist.numpy())[1])
                match_dist = torch.take_along_dim(centroid_dist, match_idx[:, None], dim=-1)
                if (match_dist < self.cfg.global_match_valid_thres).sum() != 0:  # Valid transformation with inliers
                    inst_match_idx = torch.ones_like(match_idx) * -1
                    inst_match_idx[(match_dist < self.cfg.global_match_valid_thres).reshape(-1)] = match_idx[(match_dist < self.cfg.global_match_valid_thres).reshape(-1)]
                    inst_match_list.append(inst_match_idx.to(self.device))
                    valid_transform = {}
                    valid_transform['A'] = transform['A'].float().to(self.device)
                    valid_transform['T'] = transform['T'].float().to(self.device)
                    valid_transform_list.append(valid_transform)

            # Finally keep transforms with the largest number of inlier matches
            if len(inst_match_list) != 0:
                max_num_inlier_obj = max([(inst_match != -1).sum().item() for inst_match in inst_match_list])
            else:
                max_num_inlier_obj = 0
            filtered_transform_list = []
            filtered_inst_match_list = []

            for transform, inst_match in zip(valid_transform_list, inst_match_list):
                if (inst_match != -1).sum().item() == max_num_inlier_obj:
                    filtered_transform_list.append(transform)
                    filtered_inst_match_list.append(inst_match)

            full_transform_list.append(filtered_transform_list)
            full_inst_match_list.append(filtered_inst_match_list)

        return full_transform_list, full_inst_match_list

    def evaluate_pose(self, transform: Dict[str, torch.Tensor], tgt_query_points: torch.Tensor, tgt_feats: torch.Tensor, ref_query_points: torch.Tensor, ref_feats: torch.Tensor):
        A_eval = transform['A'].to(self.device).float()
        T_eval = transform['T'].to(self.device).float()

        warp_query = tgt_query_points @ A_eval.T + T_eval
        warp_feats = idw_nn_interpolation(warp_query, ref_query_points, ref_feats, dist_pow=1., agg_k=5)

        cost = self.cost_func(tgt_feats, warp_feats)
        return cost.item()

    def optimize_affine(self, scene_idx, tgt_query_points: torch.Tensor, tgt_feats: torch.Tensor, ref_query_points: torch.Tensor, ref_feats: torch.Tensor, transform_list: List[Dict[str, torch.Tensor]], return_cost: bool = True):
        # NOTE 1: We assume a transformation of form X_ref = (X_tgt - centroid_tgt) @ A.T + centroid_ref where X is of shape (N, 3)
        # NOTE 2: Thus the final transform equation is as follows, X_ref = X_tgt @ A.T + T
        # NOTE 3: We solve for 3D affine transform and 3D shifting
        # NOTE 4: A single transform is found for each centroid match
        cost_list = []
        new_transform_list = []

        for t_idx, transform in enumerate(tqdm(transform_list, desc=f"Global Matching (Scene {scene_idx + 1}, {len(transform_list)} Transforms)")):
            A_opt = transform['A'].to(self.device).float().requires_grad_()
            T_opt = transform['T'].to(self.device).float().requires_grad_()

            optimizer = torch.optim.Adam([A_opt, T_opt], self.cfg.global_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.cfg.global_patience, factor=self.cfg.global_factor)
            for it in range(self.cfg.global_num_iter):
                optimizer.zero_grad()
                warp_query = tgt_query_points @ A_opt.T + T_opt
                warp_feats = idw_nn_interpolation(warp_query, ref_query_points, ref_feats, dist_pow=1., agg_k=5)

                cost = self.cost_func(tgt_feats, warp_feats)
                cost.backward()

                if A_opt.grad is not None and T_opt.grad is not None and (torch.isnan(A_opt.grad).sum() + torch.isnan(T_opt.grad).sum()> 0):  # Prevent NaN loss
                    break

                optimizer.step()
                scheduler.step(cost)

            new_transform = {}
            new_transform['A'] = A_opt.detach().cpu()
            new_transform['T'] = T_opt.detach().cpu()
            new_transform_list.append(new_transform)
            cost_list.append(cost.item())

        if return_cost:
            return new_transform_list, cost_list
        else:
            return new_transform_list

    def optimize_scale_rot(self, scene_idx, tgt_query_points: torch.Tensor, tgt_feats: torch.Tensor, ref_query_points: torch.Tensor, ref_feats: torch.Tensor, transform_list: List[Dict[str, torch.Tensor]], return_cost: bool = True):
        # NOTE 1: We assume a transformation of form X_ref = (X_tgt - centroid_tgt) @ A.T + centroid_ref where X is of shape (N, 3)
        # NOTE 2: Thus the final transform equation is as follows, X_ref = X_tgt @ A.T + T
        # NOTE 3: We solve for 3D affine transform and 3D shifting
        # NOTE 4: A single transform is found for each centroid match
        cost_list = []
        new_transform_list = []

        for t_idx, transform in enumerate(tqdm(transform_list, desc=f"Global Matching (Scene {scene_idx + 1}, {len(transform_list)} Transforms)")):
            R_init = transform['R'].to(self.device).float()
            flip_sign = torch.det(R_init).sign()
            flip_mtx = torch.eye(3).to(self.device)
            if flip_sign.item() < 0.:
                flip_mtx[0, 0] = -1.
            yaw_opt = rot_mtx2yaw(R_init @ flip_mtx).requires_grad_()
            R_opt = yaw2rot_mtx(yaw_opt)[0]
            T_opt = transform['T'].to(self.device).float().requires_grad_()
            s_opt = (torch.diag(transform['S'].to(self.device))).requires_grad_()
            A_opt = R_opt @ flip_mtx @ torch.diag(s_opt)

            optimizer = torch.optim.Adam([yaw_opt, T_opt, s_opt], self.cfg.global_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.cfg.global_patience, factor=self.cfg.global_factor)
            for it in range(self.cfg.global_num_iter):
                optimizer.zero_grad()
                R_opt = yaw2rot_mtx(yaw_opt)[0]
                A_opt = R_opt @ flip_mtx @ torch.diag(s_opt)
                warp_query = tgt_query_points @ A_opt.T + T_opt
                warp_feats = idw_nn_interpolation(warp_query, ref_query_points, ref_feats, dist_pow=1., agg_k=5)

                cost = self.cost_func(tgt_feats, warp_feats)
                cost.backward()

                if yaw_opt.grad is not None and T_opt.grad is not None and s_opt.grad is not None and (torch.isnan(yaw_opt.grad).sum() + torch.isnan(T_opt.grad).sum() + torch.isnan(s_opt.grad).sum()> 0):  # Prevent NaN loss
                    break

                optimizer.step()
                scheduler.step(cost)

            new_transform = {}
            new_transform['A'] = A_opt.detach().cpu()
            new_transform['T'] = T_opt.detach().cpu()
            new_transform_list.append(new_transform)
            cost_list.append(cost.item())

        if return_cost:
            return new_transform_list, cost_list
        else:
            return new_transform_list

    def bbox_non_max_suppresion(self, bbox_tgt: np.array, transform_list: List[Dict[str, torch.Tensor]], cost_list: List[float]):
        # NMS code adapted from https://github.com/amusi/Non-Maximum-Suppression
        nms_threshold = self.cfg.global_nms_thres
        if len(transform_list) == 0:
            return [], []

        # bbox_tgt is a numpy array of shape (8, 3) containing 3D bounding box corners
        cost = np.array(cost_list)

        # Sort by feature cost of transforms
        order = np.argsort(cost)

        # Picked transforms
        picked_transforms = []
        picked_score = []

        # Extract bbox transforms
        transform_bbox_list = [(bbox_tgt @ transform['A'].T + transform['T']).numpy() for transform in transform_list]
        transform_bbox_list = [bbox[[0, 1, 5, 4], :][:, [0, 2]] for bbox in transform_bbox_list]  # Keep only 2D bboxes for IoU computation (index order is to ensure loop)

        # Iterate over transforms until all initial estimates are explained
        while order.size > 0:
            # The index of smallest cost
            index = order[0]

            # Pick the bounding box with largest cost
            picked_transforms.append(transform_list[index])
            picked_score.append(cost_list[index])

            # Compute IoU between current transformed bbox and remaining transformed bboxes
            curr_bbox = transform_bbox_list[index]
            ratio_list = [compute_bbox_iou(curr_bbox, transform_bbox_list[idx]) for idx in order]
            ratio = np.array(ratio_list)

            left = np.where(ratio < nms_threshold)
            order = order[left]

        return picked_transforms, picked_score

    def transform_non_max_suppresion(self, transform_list: List[Dict[str, torch.Tensor]], cost_list: List[float]):
        # NMS code adapted from https://github.com/amusi/Non-Maximum-Suppression
        nms_threshold = self.cfg.global_nms_thres
        if len(transform_list) == 0:
            return [], []

        cost = np.array(cost_list)

        # Sort by feature cost of transforms
        order = np.argsort(cost)

        # Picked transforms
        picked_transforms = []
        picked_score = []

        # Iterate over transforms until all initial estimates are explained
        while order.size > 0:
            # The index of smallest cost
            index = order[0]

            # Pick the bounding box with largest cost
            picked_transforms.append(transform_list[index])
            picked_score.append(cost_list[index])

            # Compute IoU between current transformed bbox and remaining transformed bboxes
            curr_transform = transform_list[index]
            dist_list = [compute_transform_distance(curr_transform, transform_list[idx]) for idx in order]
            dist = np.array(dist_list)

            left = np.where(dist > nms_threshold)
            order = order[left]

        return picked_transforms, picked_score


def compute_bbox_iou(bbox_0: np.array, bbox_1: np.array):
    # bbox_* are of shape (4, 2) and not assumed to be axis-aligned
    poly_0 = Polygon(bbox_0)
    poly_1 = Polygon(bbox_1)
    intersection_area = poly_0.intersection(poly_1).area
    union_area = poly_0.union(poly_1).area

    if union_area != 0.:
        return intersection_area / union_area
    else:
        return 0.


def compute_transform_distance(transform_0: Dict[str, torch.Tensor], transform_1: Dict[str, torch.Tensor]):
    A_dist = (transform_0['A'] - transform_1['A']).abs().max()
    T_dist = (transform_0['T'] - transform_1['T']).abs().max()
    return A_dist + T_dist
