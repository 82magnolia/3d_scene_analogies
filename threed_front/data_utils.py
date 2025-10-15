from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
from threed_front.scene_gen_utils import (
    generate_grid_points,
    generate_2d_grid_points,
    generate_grid_points_from_centroids,
    generate_uniform_query_points,
    check_in_polygon,
    fp_uniform_sample
)
from threed_front.utils import choice_without_replacement
from glob import glob
import os
import open3d as o3d
from tqdm import tqdm
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import ball_query, knn_points, knn_gather
from pytorch3d.ops.utils import masked_gather
import torch
import warnings
import random
from typing import NamedTuple
from trimesh.sample import sample_surface
from PIL import Image
import argparse
from threed_front.utils import trimesh_load_with_postprocess
import json
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
import pandas as pd


class KeypointSceneBuffer(Dataset):
    def __init__(self, scene_path_pair_list: List[Tuple[str, str]], cfg: NamedTuple, device: torch.DeviceObjType):
        # scene_path_pair_list is a list containing scenes to use for filling the buffer
        super(KeypointSceneBuffer).__init__()
        self.scene_list = []

        # Load config values
        self.cfg = cfg
        self.device = device
        self.num_sample_query = self.cfg.num_sample_query
        self.num_cache_query = self.cfg.num_cache_query
        self.query_point_type = self.cfg.query_point_type  # Type of query point sampling to apply for caching
        self.query_sampling_method = getattr(self.cfg, "query_sampling_method", "random")  # Type of sampling for query points
        self.region_scale_range = getattr(self.cfg, "region_scale_range", [0.5, 2.5])  # Minimum / maximum range values for generating sampling regions when query_sampling_method is 'region'
        num_vert_split = self.cfg.num_vert_split  # Number of points per vertical level, used for uniform floorplan sampling
        fp_feat_type = self.cfg.fp_feat_type  # Type of features to use for describing floorplan keypoints
        force_up_margin = self.cfg.force_up_margin,  # Forced upper vertical margin value for floorplan queries
        force_low_margin = self.cfg.force_low_margin,  # Forced lower vertical margin value for floorplan queries
        obj_point_query_scale_factor = self.cfg.obj_point_query_scale_factor  # List containing scale factors to use for object point queries

        # Post-process 3D-FUTURE object directory
        with open(self.cfg.obj_json, 'r') as f:
            obj_list = json.load(f)

        obj_classes = []
        cls2obj_id = {}
        obj_id2class = {}
        for obj_dict in obj_list:
            obj_class = obj_dict['super-category'].lower()
            obj_classes.append(obj_class)

            if obj_class not in cls2obj_id.keys():
                cls2obj_id[obj_class] = []
            cls2obj_id[obj_class].append(obj_dict['model_id'])
            obj_id2class[obj_dict['model_id']] = obj_class

        obj_classes = list(set(obj_classes))

        scene_names_list = ['pos', 'pair_pos', 'neg']

        # Determine scene pair types
        """
            NOTE: There are currently three types of scene pairs.

            Type 1. 'homo' pairs: Pairs generated only with object swaps
            Type 2. 'real_hetero' pairs: Pairs generated from rule-based scene-level semantics matching
            Type 3. 'synthetic_hetero' pairs: Pairs generated from synthetic manipulation of one scene to another
        """
        if any([path_0 != path_1 for (path_0, path_1) in scene_path_pair_list]):  # Heterogeneous scenes
            if all(['augment' in path_1 for (_, path_1) in scene_path_pair_list]):  # Heterogeneous scenes from synthetic augmentations
                self.scene_pair_type = 'synthetic_hetero'
            else:
                self.scene_pair_type = 'real_hetero'
        else:
            self.scene_pair_type = 'homo'

        # Initialize external object instances as None
        self.external_obj_inst = None

        # Optionally load pre-specified object instances to use for sampling object query points
        if getattr(cfg, 'obj_inst_from_transform_cache', False):
            # NOTE: This is used when we want to evaluate pre-computed transforms
            estim_transform_dict = torch.load(cfg.transform_path)
            self.external_obj_inst = {}
            for path_0, path_1 in estim_transform_dict.keys():
                self.external_obj_inst[(path_0, path_1)] = estim_transform_dict[(path_0, path_1)]['tgt_inst_labels']

        # Optionally load pre-specified object instances from external pair files
        if getattr(self.cfg, "scene_pair_type", None) == "manual":
            assert getattr(self.cfg, "scene_pair_file", None) is not None
            scene_pair_table = pd.read_table(self.cfg.scene_pair_file)
            ref_scene_path_list = [os.path.join(self.cfg.scene_root, path, "scene.npz") for path in scene_pair_table.ref]
            tgt_scene_path_list = [os.path.join(self.cfg.scene_root, path, "scene.npz") for path in scene_pair_table.tgt]
            self.test_scene_path_pair_list = [(os.path.normpath(ref_path), os.path.normpath(tgt_path)) for (ref_path, tgt_path) in zip(ref_scene_path_list, tgt_scene_path_list)]
            self.external_obj_inst = {}
            for path_idx, (ref_path, tgt_path) in enumerate(self.test_scene_path_pair_list):
                # NOTE: Reference scene is "pos" scene and target scene is "pair_pos" scene
                if eval(scene_pair_table.tgt_inst[path_idx].split(',')[0]) == -1:  # Load all instances
                    self.external_obj_inst[(ref_path, tgt_path)] = [inst_val for inst_val in range(np.load(tgt_scene_path_list[0])['pos_obj_path'].shape[0])]
                else:
                    self.external_obj_inst[(ref_path, tgt_path)] = [eval(inst_val) for inst_val in scene_pair_table.tgt_inst[path_idx].split(',')]

        print("Initializing keypoint scene buffer...")
        # Save scenes for loading: each scene contains keypoints, features, and query point locations
        for scene_idx, (scene_path_pos, scene_path_pair_pos) in enumerate(tqdm(scene_path_pair_list)):
            # Load scene information
            scene_triplet_pos = np.load(scene_path_pos)
            scene_triplet_pair_pos = np.load(scene_path_pair_pos, allow_pickle=True)  # To load 'pair_pos_idx_map'

            self.scene_list.append({})

            for scene_name in scene_names_list:
                self.scene_list[-1][scene_name] = {}
                if scene_name in ['pos', 'neg']:
                    scene_triplet = scene_triplet_pos
                    scene_path = scene_path_pos
                else:
                    scene_triplet = scene_triplet_pair_pos
                    scene_path = scene_path_pair_pos
                obj_trans = scene_triplet[scene_name + "_trans"]
                obj_rot = scene_triplet[scene_name + "_rot"]
                obj_scene_scales = scene_triplet[scene_name + "_obj_scene_scales"]

                scene_fp_pts = scene_triplet[scene_name + "_fp_points"]  # (N_fp_pts, 3)
                wireframe_fp_pts = np.copy(scene_fp_pts)
                if scene_fp_pts.shape[0] != 0:
                    if self.cfg.fp_point_type == "wall_samples":  # Sampling along walls
                        scene_fp_pts = fp_uniform_sample(scene_fp_pts, self.cfg.fp_sample_step_size[0], self.cfg.fp_sample_step_size[1], add_floor_ceil=False)
                    elif self.cfg.fp_point_type == "surface_samples":  # Sampling along walls, floors, and ceilings
                        scene_fp_pts = fp_uniform_sample(scene_fp_pts, self.cfg.fp_sample_step_size[0], self.cfg.fp_sample_step_size[1], add_floor_ceil=True)

                    if self.cfg.fp_label_type == "single":
                        scene_fp_pts = np.concatenate([scene_fp_pts, np.ones_like(scene_fp_pts[:, 0:1]) * -1], axis=-1)  # Floorplans are assigned with labels negative one
                    else:  # Additional labels for ground (label as -2) and ceiling (label as -3)
                        scene_fp_pts = np.concatenate([scene_fp_pts, np.ones_like(scene_fp_pts[:, 0:1]) * -1], axis=-1)  # Floorplans are assigned with labels negative one
                        is_ground = (scene_fp_pts[:, 1] == scene_fp_pts[:, 1].min())
                        is_ceil = (scene_fp_pts[:, 1] == scene_fp_pts[:, 1].max())
                        scene_fp_pts[is_ground, -1] = -2
                        scene_fp_pts[is_ceil, -1] = -3
                else:
                    scene_fp_pts = np.concatenate([scene_fp_pts, np.ones_like(scene_fp_pts[:, 0:1]) * -1], axis=-1)  # Floorplans are assigned with labels negative one

                scene_obj_pts = []
                scene_obj_feats = []

                # Generate object points and features
                for obj_idx in range(scene_triplet[scene_name + "_obj_id"].shape[0]):
                    sample_path = os.path.join(
                        self.cfg.point_sample_root,
                        f"sample_{self.cfg.scene_sample_points}",
                        scene_triplet[scene_name + "_obj_id"][obj_idx] + ".npy"
                    )
                    scene_sample_pcd_np = np.load(sample_path)
                    scene_sample_pcd_np *= scene_triplet[scene_name + '_obj_scene_scales'][obj_idx]
                    scene_sample_pcd_np = scene_sample_pcd_np @ scene_triplet[scene_name + '_rot'][obj_idx].T + scene_triplet[scene_name + '_trans'][obj_idx]
                    scene_sample_pcd_np = np.concatenate([scene_sample_pcd_np, np.ones_like(scene_sample_pcd_np[:, 0:1]) * obj_idx], axis=-1)
                    scene_obj_pts.append(scene_sample_pcd_np)

                # Aggregate point and instance information
                scene_obj_pts = np.concatenate(scene_obj_pts, axis=0)
                scene_pts_inst_labels = np.concatenate([scene_fp_pts, scene_obj_pts], axis=0)
                scene_pts = scene_pts_inst_labels[:, :3]

                scene_inst_labels = scene_pts_inst_labels[:, 3:].astype(int)  # (N_pts, 1)
                obj_sem_labels = scene_triplet[scene_name + "_obj_classes"][:, None]  # (N_obj, 1)

                if self.cfg.fp_label_type == "single":
                    obj_sem_labels = np.concatenate([obj_sem_labels, np.array([[-1]])], axis=0)  # (N_obj + 1, 1)
                else:
                    obj_sem_labels = np.concatenate([obj_sem_labels, np.array([[-3], [-2], [-1]])], axis=0)  # (N_obj + 3, 1)

                scene_sem_labels = obj_sem_labels.squeeze(-1)[scene_inst_labels.squeeze(-1)][:, None]  # (N_pts, 1)
                self.scene_list[-1][scene_name]["points"] = scene_pts
                self.scene_list[-1][scene_name]["instance"] = scene_inst_labels
                self.scene_list[-1][scene_name]["semantics"] = scene_sem_labels
                self.scene_list[-1][scene_name]["obj_trans"] = obj_trans
                self.scene_list[-1][scene_name]["obj_rot"] = obj_rot
                self.scene_list[-1][scene_name]["obj_scene_scales"] = obj_scene_scales
                self.scene_list[-1][scene_name]["obj_id"] = scene_triplet[scene_name + "_obj_id"]

                # Load subclasses if they exist in scene_triplet
                if scene_name + "_obj_subclasses_str" in scene_triplet.keys():
                    self.scene_list[-1][scene_name]["obj_subclasses"] = scene_triplet[scene_name + "_obj_subclasses_str"]

                # Assign object matching instance labels used for group-based query point sampling
                if self.scene_pair_type in ['real_hetero', 'homo']:
                    self.scene_list[-1][scene_name]["obj_match_instance"] = np.arange(obj_trans.shape[0])
                else:
                    if scene_name in ['pos', 'neg']:
                        self.scene_list[-1][scene_name]["obj_match_instance"] = np.arange(obj_trans.shape[0])
                    else:
                        obj_idx_map = scene_triplet['pair_pos_idx_map'].item()
                        self.scene_list[-1][scene_name]["obj_match_instance"] = np.array([
                            obj_idx_map[obj_idx] for obj_idx in range(obj_trans.shape[0])
                        ])

                # Set scene_path as an additional key for visualization
                self.scene_list[-1][scene_name]['scene_path'] = scene_path

                num_pts = scene_pts.shape[0]
                self.scene_list[-1][scene_name]["feats"] = np.zeros([num_pts, 0])

                scene_obj_paths = scene_triplet[scene_name + "_obj_path"]
                num_obj = scene_obj_paths.shape[0]

                # Generate query points
                if self.query_point_type == 'obj_points':  # Object-proximal points
                    num_grid_points = int((self.num_cache_query // num_obj) ** (1 / 3))
                    if obj_point_query_scale_factor is None:
                        obj_point_query_scale_factor = [1., 1., 1.]

                    if scene_name == 'neg':  # Sample from original positive object points for training
                        grid_scene_name = 'pos'
                    else:
                        grid_scene_name = scene_name

                    query_points = generate_grid_points(
                        num_grid_points,
                        scene_triplet[grid_scene_name + "_bboxes"][:, None, :],
                        scene_triplet[grid_scene_name + "_bbox_centroids"],
                        scene_triplet[grid_scene_name + "_rot"],
                        scene_triplet[grid_scene_name + "_trans"],
                        scale_factors=obj_point_query_scale_factor
                    )
                    query_inst_labels = -1 * np.ones(shape=[query_points.shape[0]])
                    query_points_colors = -1 * np.ones(shape=[query_points.shape[0], 3])
                elif self.query_point_type == 'obj_surface_points':  # Object-surface points from keypoint locations
                    query_points = np.copy(scene_pts)[scene_inst_labels.reshape(-1).astype(int) >= 0.]
                    query_inst_labels = np.copy(scene_inst_labels.reshape(-1))[scene_inst_labels.reshape(-1).astype(int) >= 0.].astype(int)
                    x_sort_idx = np.argsort(query_points[:, 0])
                    y_sort_idx = np.argsort(query_points[x_sort_idx, 1])
                    z_sort_idx = np.argsort(query_points[x_sort_idx[y_sort_idx], 2])
                    query_points = query_points[x_sort_idx[y_sort_idx[z_sort_idx]]]
                    query_inst_labels = query_inst_labels[x_sort_idx[y_sort_idx[z_sort_idx]]]
                    query_points_colors = -1 * np.ones(shape=[query_points.shape[0], 3])
                elif self.query_point_type == 'obj_sample_points':  # Object points sampled to a designated number
                    warnings.warn("For obj_sample_points num_cache_query indicates number of caching points per object")
                    query_points_list = []
                    query_inst_labels_list = []
                    query_points_colors_list = []
                    for obj_idx, obj_path in enumerate(scene_obj_paths):
                        sample_path = os.path.join(
                            self.cfg.point_sample_root,
                            f"sample_{self.num_cache_query}",
                            scene_triplet[scene_name + "_obj_id"][obj_idx] + ".npy"
                        )
                        if not os.path.exists(sample_path) or getattr(self.cfg, "vis_local_match_mode", None) == "texture_transfer":  # Load color for query points
                            obj_mesh = trimesh_load_with_postprocess(obj_path, 'bottom_crop')
                            points, _, colors = sample_surface(obj_mesh, self.num_cache_query, sample_color=True)
                            points = np.asarray(points)
                            colors = np.asarray(colors)[:, :3] / 255.

                            obj_pcd = o3d.geometry.PointCloud()
                            obj_pcd.points = o3d.utility.Vector3dVector(points)
                            obj_pcd.colors = o3d.utility.Vector3dVector(colors)

                            if len(obj_pcd.colors) == 0:
                                obj_pcd.colors = o3d.utility.Vector3dVector(
                                    np.ones_like(np.asarray(obj_pcd.points))
                                )

                            obj_pcd_np = np.asarray(obj_pcd.points)
                            obj_rgb_np = np.asarray(obj_pcd.colors)

                            obj_pcd_np = obj_pcd_np * obj_scene_scales[obj_idx: obj_idx + 1]
                            obj_pcd_np = obj_pcd_np @ obj_rot[obj_idx].T
                            obj_pcd_np += obj_trans[obj_idx]

                            query_points_list.append(obj_pcd_np)
                            query_inst_labels_list.append(np.ones([obj_pcd_np.shape[0], ], dtype=int) * obj_idx)
                            query_points_colors_list.append(obj_rgb_np)
                        else:
                            obj_pcd_np = np.load(sample_path)
                            obj_pcd_np *= obj_scene_scales[obj_idx: obj_idx + 1]
                            obj_pcd_np = obj_pcd_np @ obj_rot[obj_idx].T + obj_trans[obj_idx]

                            query_points_list.append(obj_pcd_np)
                            query_inst_labels_list.append(np.ones([obj_pcd_np.shape[0], ], dtype=int) * obj_idx)
                            query_points_colors_list.append(-1 * np.ones(shape=[obj_pcd_np.shape[0], 3]))
                    query_points = np.concatenate(query_points_list, axis=0)
                    query_inst_labels = np.concatenate(query_inst_labels_list, axis=0)
                    query_points_colors = np.concatenate(query_points_colors_list, axis=0)

                    x_sort_idx = np.argsort(query_points[:, 0])
                    y_sort_idx = np.argsort(query_points[x_sort_idx, 1])
                    z_sort_idx = np.argsort(query_points[x_sort_idx[y_sort_idx], 2])
                    query_points = query_points[x_sort_idx[y_sort_idx[z_sort_idx]]]
                    query_inst_labels = query_inst_labels[x_sort_idx[y_sort_idx[z_sort_idx]]]
                    query_points_colors = query_points_colors[x_sort_idx[y_sort_idx[z_sort_idx]]]
                elif self.query_point_type == 'obj_match_points':  # Object surface points matched from nearest neighbor alignment
                    assert self.cfg.point_sample_root is not None
                    assert not self.scene_pair_type == 'real_hetero'  # Heterogeneous scenes from real pairs are not supported
                    if scene_name == 'pos':  # Initialize query point buffer
                        query_sample_buffer = {'pos': [], 'pair_pos': []}

                    if scene_name in ['pos', 'pair_pos']:  # Load objects
                        for obj_idx, obj_path in enumerate(scene_obj_paths):
                            sample_path = os.path.join(
                                self.cfg.point_sample_root,
                                f"sample_{self.num_cache_query}",
                                scene_triplet[scene_name + "_obj_id"][obj_idx] + ".npy"
                            )
                            scene_sample_pcd_np = np.load(sample_path)
                            query_sample_buffer[scene_name].append(scene_sample_pcd_np)

                    # Setup query points
                    query_points_list = []
                    query_inst_labels_list = []
                    if scene_name in ['pos', 'neg']:  # Sample from original positive object points for training
                        for obj_idx, obj_path in enumerate(scene_obj_paths):
                            query_points = query_sample_buffer['pos'][obj_idx]

                            query_points = query_points * obj_scene_scales[obj_idx: obj_idx + 1]
                            query_points = query_points @ obj_rot[obj_idx].T
                            query_points += obj_trans[obj_idx]
                            query_points_list.append(query_points)
                            query_inst_labels_list.append(np.ones([query_points.shape[0], ], dtype=int) * obj_idx)
                    else:  # Perform query point re-ordering from NN matching for pair_pos points
                        if self.scene_pair_type == 'homo':  # Directly align matching object pairs
                            for obj_idx, obj_path in enumerate(scene_obj_paths):
                                # Scale points for obtaining obtimal alignments
                                query_points = query_sample_buffer['pair_pos'][obj_idx]
                                query_points = query_points * obj_scene_scales[obj_idx: obj_idx + 1]

                                ref_query_points = query_sample_buffer['pos'][obj_idx]
                                ref_query_points = ref_query_points * scene_triplet["pos_obj_scene_scales"][obj_idx: obj_idx + 1]
                                query_dist = np.linalg.norm(ref_query_points[:, None, :] - query_points[None, :, :], axis=-1)

                                if self.cfg.query_obj_match_mode == "jv":
                                    _, match_col_ind = linear_sum_assignment(query_dist)  # Optimal assignment from Hungarian matching
                                    query_points = query_points[match_col_ind, :]
                                else:
                                    query_points = query_points[query_dist.argmin(-1)]

                                # Perform remaining transformation prior to adding
                                query_points = query_points @ obj_rot[obj_idx].T
                                query_points += obj_trans[obj_idx]

                                query_points_list.append(query_points)
                                query_inst_labels_list.append(np.ones([query_points.shape[0], ], dtype=int) * obj_idx)
                        else:  # Align object pairs from synthetic augmentations
                            # NOTE: We assume 'pos' scene is augmented with additional objects and floorplan perturbations to produce 'pair_pos'
                            obj_idx_map = scene_triplet['pair_pos_idx_map'].item()  # Dictionary mapping object indices in pair_pos to corresponding objects in pos
                            for obj_idx, obj_path in enumerate(scene_obj_paths):
                                query_points = query_sample_buffer['pair_pos'][obj_idx]
                                query_points = query_points * obj_scene_scales[obj_idx: obj_idx + 1]

                                match_obj_idx = obj_idx_map[obj_idx]
                                if match_obj_idx == -1:  # No matching instance in 'pos' for object in 'pair_pos'
                                    continue
                                ref_query_points = query_sample_buffer['pos'][match_obj_idx]
                                ref_query_points = ref_query_points * scene_triplet_pos["pos_obj_scene_scales"][match_obj_idx: match_obj_idx + 1]
                                query_dist = np.linalg.norm(ref_query_points[:, None, :] - query_points[None, :, :], axis=-1)

                                if self.cfg.query_obj_match_mode == "jv":
                                    _, match_col_ind = linear_sum_assignment(query_dist)  # Optimal assignment from Hungarian matching
                                    query_points = query_points[match_col_ind, :]
                                else:
                                    query_points = query_points[query_dist.argmin(-1)]

                                query_points = query_points @ obj_rot[obj_idx].T
                                query_points += obj_trans[obj_idx]

                                query_points_list.append(query_points)
                                query_inst_labels_list.append(np.ones([query_points.shape[0], ], dtype=int) * obj_idx)

                    if self.cfg.query_obj_match_add_bbox:
                        num_grid_points = int((self.num_cache_query) ** (1 / 3))  # NOTE: We want to have same number of points per object
                        if obj_point_query_scale_factor is None:
                            obj_point_query_scale_factor = [1., 1., 1.]

                        if scene_name == 'neg':  # Sample from original positive object points for training
                            grid_scene_name = 'pos'
                        else:
                            grid_scene_name = scene_name

                        query_points = generate_grid_points(
                            num_grid_points,
                            scene_triplet[grid_scene_name + "_bboxes"][:, None, :],
                            scene_triplet[grid_scene_name + "_bbox_centroids"],
                            scene_triplet[grid_scene_name + "_rot"],
                            scene_triplet[grid_scene_name + "_trans"],
                            scale_factors=obj_point_query_scale_factor
                        )
                        query_points_list.append(query_points)
                        query_inst_labels_list.append(np.ones([query_points.shape[0], ], dtype=int) * -1)
                    query_points = np.concatenate(query_points_list, axis=0)
                    query_inst_labels = np.concatenate(query_inst_labels_list, axis=0)
                    query_points_colors = -1 * np.ones(shape=[query_points.shape[0], 3])
                elif self.query_point_type == 'obj_points_planar':  # Object-proximal points for sampled in a co-planar fashion
                    num_grid_points = int((self.num_cache_query // num_obj) ** (1 / 2))
                    if obj_point_query_scale_factor is None:
                        obj_point_query_scale_factor = [1., 1.]

                    if scene_name == 'neg':  # Sample from original positive object points for training
                        grid_scene_name = 'pos'
                    else:
                        grid_scene_name = scene_name
                    query_points = generate_2d_grid_points(
                        num_grid_points,
                        scene_triplet[grid_scene_name + "_bbox_centroids"][:, [0, 2]],
                        scene_triplet[grid_scene_name + "_bboxes"][:, None, [0, 2]],
                        scene_triplet[grid_scene_name + "_rot"][:, [0, 2], :][:, :, [0, 2]],
                        scene_triplet[grid_scene_name + "_trans"][:, [0, 2]],
                        scale_factors=obj_point_query_scale_factor
                    )

                    # Add vertical points
                    fp_kpts = scene_pts[scene_inst_labels.reshape(-1) == -1]
                    max_vert_level = fp_kpts[:, 1].max() if force_up_margin is None else force_up_margin
                    min_vert_level = fp_kpts[:, 1].min() if force_low_margin is None else force_low_margin
                    vert_points = np.linspace(min_vert_level, max_vert_level, num_vert_split)
                    vert_points = np.repeat(vert_points, query_points.shape[0])[:, None]

                    # Convert coordinate frames to match scene frame
                    query_points = np.concatenate([query_points[:, 0:1], vert_points, query_points[:, 1:2]], axis=-1)
                    query_inst_labels = -1 * np.ones(shape=[query_points.shape[0]])
                    query_points_colors = -1 * np.ones(shape=[query_points.shape[0], 3])
                elif self.query_point_type == 'floorplan_points':  # Floorplan interior points
                    scene_min_x = wireframe_fp_pts[:, 0].min()
                    scene_max_x = wireframe_fp_pts[:, 0].max()
                    scene_min_y = wireframe_fp_pts[:, 2].min()
                    scene_max_y = wireframe_fp_pts[:, 2].max()
                    init_num_query = 1000  # Used for determining area ratio between floorplan & bounding box
                    init_query_points = generate_uniform_query_points(init_num_query, scene_min_x, scene_max_x, scene_min_y, scene_max_y)
                    init_in_polygon = check_in_polygon(wireframe_fp_pts[:wireframe_fp_pts.shape[0] // 2, [0, 2]], init_query_points)
                    area_ratio = init_in_polygon.sum() / init_in_polygon.shape[0]
                    num_surplus = self.num_cache_query / (area_ratio * num_vert_split)  # Discount for area_ratio when generating query points
                    surplus_query_points = generate_uniform_query_points(num_surplus, scene_min_x, scene_max_x, scene_min_y, scene_max_y)
                    surplus_in_polygon = check_in_polygon(wireframe_fp_pts[:wireframe_fp_pts.shape[0] // 2, [0, 2]], surplus_query_points)
                    query_points = surplus_query_points[surplus_in_polygon]

                    # Add vertical points
                    max_vert_level = wireframe_fp_pts[:, 1].max() if force_up_margin is None else wireframe_fp_pts[:, 1].max() - force_up_margin
                    min_vert_level = wireframe_fp_pts[:, 1].min() if force_low_margin is None else wireframe_fp_pts[:, 1].min() + force_low_margin
                    vert_points = np.linspace(min_vert_level, max_vert_level, num_vert_split)
                    vert_points = np.repeat(vert_points, query_points.shape[0])[:, None]
                    query_points = np.tile(query_points, (num_vert_split, 1))
                    query_points = np.concatenate([query_points[:, 0:1], vert_points, query_points[:, 1:2]], axis=-1)
                    query_inst_labels = -1 * np.ones(shape=[query_points.shape[0]])
                    query_points_colors = -1 * np.ones(shape=[query_points.shape[0], 3])
                elif self.query_point_type == 'floor_points':  # Floor points extracted from floorplan bounding box
                    scene_min_x = wireframe_fp_pts[:, 0].min()
                    scene_max_x = wireframe_fp_pts[:, 0].max()
                    scene_min_y = wireframe_fp_pts[:, 2].min()
                    scene_max_y = wireframe_fp_pts[:, 2].max()
                    query_points = generate_uniform_query_points(self.num_cache_query // num_vert_split, scene_min_x, scene_max_x, scene_min_y, scene_max_y)

                    # Add vertical points
                    max_vert_level = wireframe_fp_pts[:, 1].max() if force_up_margin is None else wireframe_fp_pts[:, 1].max() - force_up_margin
                    min_vert_level = wireframe_fp_pts[:, 1].min() if force_low_margin is None else wireframe_fp_pts[:, 1].min() + force_low_margin
                    vert_points = np.linspace(min_vert_level, max_vert_level, num_vert_split)
                    vert_points = np.repeat(vert_points, query_points.shape[0])[:, None]
                    query_points = np.tile(query_points, (num_vert_split, 1))
                    query_points = np.concatenate([query_points[:, 0:1], vert_points, query_points[:, 1:2]], axis=-1)
                    query_inst_labels = -1 * np.ones(shape=[query_points.shape[0]])
                    query_points_colors = -1 * np.ones(shape=[query_points.shape[0], 3])
                else:
                    raise ValueError("Invalid query point type")

                self.scene_list[-1][scene_name]["query_points"] = query_points
                self.scene_list[-1][scene_name]["query_inst_labels"] = query_inst_labels
                self.scene_list[-1][scene_name]["query_points_colors"] = query_points_colors

    def __getitem__(self, idx):
        pos_scene = {k: v for k, v in self.scene_list[idx]["pos"].items()}
        pair_pos_scene = {k: v for k, v in self.scene_list[idx]["pair_pos"].items()}
        neg_scene = {k: v for k, v in self.scene_list[idx]["neg"].items()}

        if self.scene_pair_type in ['real_hetero', 'synthetic_hetero']:  # Only a fixed set of sampling methods are supported for heterogeneous scenes 
            assert self.query_sampling_method in ["region", "region_2d", "random_instance", "random_group"]

        if getattr(self.cfg, "enforce_sampling", False) or self.num_sample_query != self.num_cache_query:  # Skip sampling if num_sample_query and num_cache_query is the same
            if self.scene_pair_type in ['real_hetero', 'synthetic_hetero']:
                # NOTE 1: All random sampling is performed around pair_pos which is used as "reference" during field matching
                # NOTE 2: Query points in other types of scenes (pos, neg) are all dummy points
                if self.query_sampling_method in ["region", "region_2d"]:  # Choose one centroid point and sample a cuboid region around it
                    pair_pos_region_centroid, query_idx = choice_without_replacement(
                        self.scene_list[idx]["pair_pos"]["query_points"],
                        1,
                        return_idx=True
                    )  # (1, 3)
                    pos_region_centroid = pos_scene["query_points"][0]  # (1, 3)
                    neg_region_centroid = neg_scene["query_points"][0]  # (1, 3)

                    x_range = random.uniform(self.region_scale_range[0], self.region_scale_range[1])
                    y_range = random.uniform(self.region_scale_range[0], self.region_scale_range[1]) if self.query_sampling_method == "region" else 0.  # This corresponds to height dimension
                    z_range = random.uniform(self.region_scale_range[0], self.region_scale_range[1])

                    # Ensure y_range generates points above ground level
                    y_range = min(2 * pair_pos_region_centroid[0, 1].item(), y_range)

                    num_point_per_axis = int(self.num_sample_query ** (1 / 3)) if self.query_sampling_method == "region" else int(self.num_sample_query ** (1 / 2))
                    pair_pos_scene["query_points"] = generate_grid_points_from_centroids(num_point_per_axis, pair_pos_region_centroid, (x_range, y_range, z_range)).reshape(-1, 3)
                    pair_pos_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                    pair_pos_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1

                    # Set dummy points for both real and synthetic pairs
                    pos_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                    neg_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                    pos_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                    neg_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                    pos_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                    neg_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1

                elif self.query_sampling_method == "random_instance":
                    assert self.scene_list[idx]["pos"]["query_inst_labels"] is not None
                    # Identify valid instance labels that are not -1
                    match_inst_labels = pair_pos_scene["obj_match_instance"]  # Instance idx of pos scene co-occuring in pair_pos scene
                    pair_pos_inst_labels = self.scene_list[idx]["pair_pos"]["query_inst_labels"]

                    sample_inst = random.choice([k for k, v in enumerate(match_inst_labels) if v != -1])
                    pair_pos_inst_mask = (pair_pos_inst_labels == sample_inst)
                    pair_pos_scene["query_points"] = self.scene_list[idx]["pair_pos"]["query_points"][pair_pos_inst_mask]
                    pair_pos_scene["query_inst_labels"] = self.scene_list[idx]["pair_pos"]["query_inst_labels"][pair_pos_inst_mask, None]
                    pair_pos_scene["query_points_colors"] = self.scene_list[idx]["pair_pos"]["query_points_colors"][pair_pos_inst_mask]

                    if self.scene_pair_type == 'real_hetero':  # Set dummy points
                        pos_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        neg_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        pos_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                        neg_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                        pos_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        neg_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1                    
                    else:  # Set matching points for synthetic_hetero
                        pos_inst_labels = self.scene_list[idx]["pos"]["query_inst_labels"]
                        pos_inst_mask = (pos_inst_labels == match_inst_labels[sample_inst])
                        pos_scene["query_points"] = self.scene_list[idx]["pos"]["query_points"][pos_inst_mask]
                        pos_scene["query_inst_labels"] = self.scene_list[idx]["pos"]["query_inst_labels"][pos_inst_mask, None]
                        pos_scene["query_points_colors"] = self.scene_list[idx]["pos"]["query_points_colors"][pos_inst_mask]

                        neg_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        neg_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                        neg_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1

                elif self.query_sampling_method in ["random_group"]:
                    assert self.scene_list[idx]["pos"]["query_inst_labels"] is not None
                    match_inst_labels = pair_pos_scene["obj_match_instance"]  # Instance idx of pos scene co-occuring in pair_pos scene
                    if self.external_obj_inst is None:  # Generate scene samples randomly
                        # Take K nearest neighbors to identify object groups
                        valid_inst_mask = (match_inst_labels != -1)
                        valid_inst_idx = np.where(valid_inst_mask)[0]

                        obj_centroids = pair_pos_scene["obj_trans"][valid_inst_mask]  # pair_pos is used since this is the "target" scene to warp for field matching
                        obj_dist_mtx = np.linalg.norm(obj_centroids[:, None, :] - obj_centroids[None, :, :], axis=-1)
                        obj_dist_mtx[np.diag_indices(obj_dist_mtx.shape[0])] = np.inf

                        # Choose seed object and select top-K neighbors
                        topk = random.randint(self.cfg.random_group_topk_range[0], self.cfg.random_group_topk_range[1])
                        topk = min(obj_dist_mtx.shape[0] - 1, topk)
                        seed_obj_idx = np.random.choice(np.arange(obj_dist_mtx.shape[0]))
                        topk_idx = np.argpartition(obj_dist_mtx, kth=topk, axis=-1)[seed_obj_idx, :topk]
                        select_idx_list = [valid_inst_idx[seed_obj_idx]] + valid_inst_idx[topk_idx].tolist()
                    else:  # Generate scene samples from pre-computed instance list (used for evaluating transforms in main_eval_transfrom.py)
                        select_idx_list = self.external_obj_inst[
                            (
                                os.path.normpath(self.scene_list[idx]["pos"]["scene_path"]),
                                os.path.normpath(self.scene_list[idx]["pair_pos"]["scene_path"])
                            )
                        ]

                    # NOTE: We can't use the same masking strategy as 'homogenous' scenes since object orders are not the same
                    pair_pos_inst_idx = np.concatenate([
                        np.where(self.scene_list[idx]["pair_pos"]["query_inst_labels"] == select_idx)[0] for select_idx in select_idx_list
                    ])

                    pair_pos_scene["query_points"] = self.scene_list[idx]["pair_pos"]["query_points"][pair_pos_inst_idx]
                    pair_pos_scene["query_inst_labels"] = self.scene_list[idx]["pair_pos"]["query_inst_labels"][pair_pos_inst_idx, None]
                    pair_pos_scene["query_points_colors"] = self.scene_list[idx]["pair_pos"]["query_points_colors"][pair_pos_inst_idx]

                    if self.scene_pair_type == 'real_hetero':  # Set dummy points
                        pos_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        neg_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        pos_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                        neg_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                        pos_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        neg_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                    else:  # Set matching points for synthetic_hetero
                        pos_inst_idx = np.concatenate([
                            np.where(self.scene_list[idx]["pos"]["query_inst_labels"] == match_inst_labels[select_idx])[0] for select_idx in select_idx_list
                        ])

                        pos_scene["query_points"] = self.scene_list[idx]["pos"]["query_points"][pos_inst_idx]
                        pos_scene["query_inst_labels"] = self.scene_list[idx]["pos"]["query_inst_labels"][pos_inst_idx, None]
                        pos_scene["query_points_colors"] = self.scene_list[idx]["pos"]["query_points_colors"][pos_inst_idx]

                        neg_scene["query_points"] = np.ones_like(pair_pos_scene["query_points"]) * -1
                        neg_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
                        neg_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1

                else:
                    raise NotImplementedError("Other query point sampling types not supported")
            else:
                if self.query_sampling_method == "random":
                    pos_scene["query_points"], query_idx = choice_without_replacement(
                        self.scene_list[idx]["pos"]["query_points"],
                        self.num_sample_query,
                        return_idx=True
                    )
                    pair_pos_scene["query_points"] = pair_pos_scene["query_points"][query_idx]
                    neg_scene["query_points"] = neg_scene["query_points"][query_idx]
                    pos_scene["query_inst_labels"] = self.scene_list[idx]["pos"]["query_inst_labels"][query_idx, None]
                    pair_pos_scene["query_inst_labels"] = self.scene_list[idx]["pair_pos"]["query_inst_labels"][query_idx, None]
                    neg_scene["query_inst_labels"] = self.scene_list[idx]["neg"]["query_inst_labels"][query_idx, None]
                    pos_scene["query_points_colors"] = self.scene_list[idx]["pos"]["query_points_colors"][query_idx]
                    pair_pos_scene["query_points_colors"] = self.scene_list[idx]["pair_pos"]["query_points_colors"][query_idx]
                    neg_scene["query_points_colors"] = self.scene_list[idx]["neg"]["query_points_colors"][query_idx]
                elif self.query_sampling_method in ["region", "region_2d"]:  # Choose one centroid point and sample a cuboid region around it
                    pos_region_centroid, query_idx = choice_without_replacement(
                        self.scene_list[idx]["pos"]["query_points"],
                        1,
                        return_idx=True
                    )  # (1, 3)
                    pair_pos_region_centroid = pair_pos_scene["query_points"][query_idx]  # (1, 3)
                    neg_region_centroid = neg_scene["query_points"][query_idx]  # (1, 3)

                    x_range = random.uniform(self.region_scale_range[0], self.region_scale_range[1])
                    y_range = random.uniform(self.region_scale_range[0], self.region_scale_range[1]) if self.query_sampling_method == "region" else 0.  # This corresponds to height dimension
                    z_range = random.uniform(self.region_scale_range[0], self.region_scale_range[1])

                    # Ensure y_range generates points above ground level
                    min_y = min(pos_region_centroid[0, 1].item(), pair_pos_region_centroid[0, 1].item(), neg_region_centroid[0, 1].item())
                    y_range = min(2 * min_y, y_range)

                    num_point_per_axis = int(self.num_sample_query ** (1 / 3)) if self.query_sampling_method == "region" else int(self.num_sample_query ** (1 / 2))
                    pos_scene["query_points"] = generate_grid_points_from_centroids(num_point_per_axis, pos_region_centroid, (x_range, y_range, z_range)).reshape(-1, 3)
                    pair_pos_scene["query_points"] = generate_grid_points_from_centroids(num_point_per_axis, pair_pos_region_centroid, (x_range, y_range, z_range)).reshape(-1, 3)
                    neg_scene["query_points"] = generate_grid_points_from_centroids(num_point_per_axis, neg_region_centroid, (x_range, y_range, z_range)).reshape(-1, 3)
                    pos_scene["query_inst_labels"] = np.ones_like(pos_scene["query_points"][:, 0:1]) * -1
                    pair_pos_scene["query_inst_labels"] = np.ones_like(pos_scene["query_points"][:, 0:1]) * -1
                    neg_scene["query_inst_labels"] = np.ones_like(pos_scene["query_points"][:, 0:1]) * -1
                    pos_scene["query_points_colors"] = np.ones_like(pos_scene["query_points"]) * -1
                    pair_pos_scene["query_points_colors"] = np.ones_like(pos_scene["query_points"]) * -1
                    neg_scene["query_points_colors"] = np.ones_like(pos_scene["query_points"]) * -1
                elif self.query_sampling_method == "random_instance":
                    assert self.scene_list[idx]["pos"]["query_inst_labels"] is not None
                    sample_inst = np.random.choice(self.scene_list[idx]["pos"]["query_inst_labels"], size=1)
                    pos_inst_mask = (self.scene_list[idx]["pos"]["query_inst_labels"] == sample_inst)
                    pair_pos_inst_mask = (self.scene_list[idx]["pair_pos"]["query_inst_labels"] == sample_inst)
                    neg_inst_mask = (self.scene_list[idx]["neg"]["query_inst_labels"] == sample_inst)
                    pos_scene["query_points"] = self.scene_list[idx]["pos"]["query_points"][pos_inst_mask]
                    pair_pos_scene["query_points"] = self.scene_list[idx]["pair_pos"]["query_points"][pair_pos_inst_mask]
                    neg_scene["query_points"] = self.scene_list[idx]["neg"]["query_points"][neg_inst_mask]
                    pos_scene["query_inst_labels"] = self.scene_list[idx]["pos"]["query_inst_labels"][pos_inst_mask, None]
                    pair_pos_scene["query_inst_labels"] = self.scene_list[idx]["pair_pos"]["query_inst_labels"][pair_pos_inst_mask, None]
                    neg_scene["query_inst_labels"] = self.scene_list[idx]["neg"]["query_inst_labels"][neg_inst_mask, None]
                    pos_scene["query_points_colors"] = self.scene_list[idx]["pos"]["query_points_colors"][pos_inst_mask]
                    pair_pos_scene["query_points_colors"] = self.scene_list[idx]["pair_pos"]["query_points_colors"][pair_pos_inst_mask]
                    neg_scene["query_points_colors"] = self.scene_list[idx]["neg"]["query_points_colors"][neg_inst_mask]
                elif self.query_sampling_method in ["random_group"]:
                    assert self.scene_list[idx]["pos"]["query_inst_labels"] is not None
                    # Take K nearest neighbors to identify object groups
                    obj_centroids = pair_pos_scene["obj_trans"]  # pair_pos is used since this is the "target" scene to warp for field matching
                    obj_dist_mtx = np.linalg.norm(obj_centroids[:, None, :] - obj_centroids[None, :, :], axis=-1)
                    obj_dist_mtx[np.diag_indices(obj_dist_mtx.shape[0])] = np.inf

                    # Choose seed object and select top-K neighbors
                    topk = random.randint(self.cfg.random_group_topk_range[0], self.cfg.random_group_topk_range[1])
                    topk = min(obj_dist_mtx.shape[0] - 1, topk)
                    seed_obj_idx = np.random.choice(np.arange(obj_dist_mtx.shape[0]))
                    topk_idx = np.argpartition(obj_dist_mtx, kth=topk, axis=-1)[seed_obj_idx, :topk].tolist()

                    if self.external_obj_inst is None:  # Generate scene samples randomly
                        select_idx_list = [seed_obj_idx] + topk_idx
                    else:  # Generate scene samples from pre-computed instance list (used for evaluating transforms in main_eval_transfrom.py)
                        select_idx_list = self.external_obj_inst[
                            (
                                os.path.normpath(self.scene_list[idx]["pos"]["scene_path"]),
                                os.path.normpath(self.scene_list[idx]["pair_pos"]["scene_path"])
                            )
                        ]

                    pos_inst_mask = np.any(
                        np.stack(
                            [self.scene_list[idx]["pos"]["query_inst_labels"] == select_idx for select_idx in select_idx_list],
                            axis=-1
                        ), axis=-1
                    )
                    pair_pos_inst_mask = np.any(
                        np.stack(
                            [self.scene_list[idx]["pair_pos"]["query_inst_labels"] == select_idx for select_idx in select_idx_list],
                            axis=-1
                        ), axis=-1
                    )
                    neg_inst_mask = np.any(
                        np.stack(
                            [self.scene_list[idx]["neg"]["query_inst_labels"] == select_idx for select_idx in select_idx_list],
                            axis=-1
                        ), axis=-1
                    )

                    pos_scene["query_points"] = self.scene_list[idx]["pos"]["query_points"][pos_inst_mask]
                    pair_pos_scene["query_points"] = self.scene_list[idx]["pair_pos"]["query_points"][pair_pos_inst_mask]
                    neg_scene["query_points"] = self.scene_list[idx]["neg"]["query_points"][neg_inst_mask]
                    pos_scene["query_inst_labels"] = self.scene_list[idx]["pos"]["query_inst_labels"][pos_inst_mask, None]
                    pair_pos_scene["query_inst_labels"] = self.scene_list[idx]["pair_pos"]["query_inst_labels"][pair_pos_inst_mask, None]
                    neg_scene["query_inst_labels"] = self.scene_list[idx]["neg"]["query_inst_labels"][neg_inst_mask, None]
                    pos_scene["query_points_colors"] = self.scene_list[idx]["pos"]["query_points_colors"][pos_inst_mask]
                    pair_pos_scene["query_points_colors"] = self.scene_list[idx]["pair_pos"]["query_points_colors"][pair_pos_inst_mask]
                    neg_scene["query_points_colors"] = self.scene_list[idx]["neg"]["query_points_colors"][neg_inst_mask]
                else:
                    raise NotImplementedError("Other query point sampling types not supported")
        else:  # Apply null values for query instance labels and colors
            pos_scene["query_inst_labels"] = np.ones_like(pos_scene["query_points"][:, 0:1]) * -1
            pair_pos_scene["query_inst_labels"] = np.ones_like(pair_pos_scene["query_points"][:, 0:1]) * -1
            neg_scene["query_inst_labels"] = np.ones_like(neg_scene["query_points"][:, 0:1]) * -1
            pos_scene["query_points_colors"] = np.ones_like(pos_scene["query_points"]) * -1
            pair_pos_scene["query_points_colors"] = np.ones_like(pair_pos_scene["query_points"]) * -1
            neg_scene["query_points_colors"] = np.ones_like(neg_scene["query_points"]) * -1

        scene_sample = {
            'pos': pos_scene,
            'pair_pos': pair_pos_scene,
            'neg': neg_scene
        }
        return scene_sample

    def __len__(self):
        return len(self.scene_list)


def keybuff_collate_fn(batch_list):
    pos_batch_dict = {'points': [], 'feats': [], 'query_points': [], 'query_feats': []}
    pair_pos_batch_dict = {'points': [], 'feats': [], 'query_points': [], 'query_feats': []}
    neg_batch_dict = {'points': [], 'feats': [], 'query_points': [], 'query_feats': []}
    scene_keys = ['points', 'query_points', 'query_inst_labels', 'query_points_colors', 'instance', 'semantics', 'feats']  # Keys are same for all scenes

    # Cache scene paths for future visualization
    scene_path_dict = {'pos': [], 'pair_pos': [], 'neg': []}
    scene_subclasses_dict = {'pos': [], 'pair_pos': [], 'neg': []}
    scene_obj_id_dict = {'pos': [], 'pair_pos': [], 'neg': []}

    for batch in batch_list:
        pos_scene = batch['pos']
        pair_pos_scene = batch['pair_pos']
        neg_scene = batch['neg']

        pos_feat = []
        pair_pos_feat = []
        neg_feat = []

        pos_query_feat = []
        pair_pos_query_feat = []
        neg_query_feat = []
        for k in scene_keys:
            if k in ['points', 'query_points']:
                pos_batch_dict[k].append(torch.tensor(pos_scene[k]).float())
                pair_pos_batch_dict[k].append(torch.tensor(pair_pos_scene[k]).float())
                neg_batch_dict[k].append(torch.tensor(neg_scene[k]).float())
            elif k in ['query_inst_labels', 'query_points_colors']:
                pos_query_feat.append(torch.tensor(pos_scene[k]).float())
                pair_pos_query_feat.append(torch.tensor(pair_pos_scene[k]).float())
                neg_query_feat.append(torch.tensor(neg_scene[k]).float())
            else:
                pos_feat.append(torch.tensor(pos_scene[k]).float())
                pair_pos_feat.append(torch.tensor(pair_pos_scene[k]).float())
                neg_feat.append(torch.tensor(neg_scene[k]).float())

        pos_feat = torch.cat(pos_feat, dim=-1)  # (N_pts, 1 + 1 + D_emb), representing 'instance' + 'semantics' + 'feats'
        pair_pos_feat = torch.cat(pair_pos_feat, dim=-1)
        neg_feat = torch.cat(neg_feat, dim=-1)
        pos_query_feat = torch.cat(pos_query_feat, dim=-1)  # (N_pts, 1 + 3), representing 'instance' + 'colors'
        pair_pos_query_feat = torch.cat(pair_pos_query_feat, dim=-1)
        neg_query_feat = torch.cat(neg_query_feat, dim=-1)

        pos_batch_dict['feats'].append(pos_feat)
        pair_pos_batch_dict['feats'].append(pair_pos_feat)
        neg_batch_dict['feats'].append(neg_feat)
        pos_batch_dict['query_feats'].append(pos_query_feat)
        pair_pos_batch_dict['query_feats'].append(pair_pos_query_feat)
        neg_batch_dict['query_feats'].append(neg_query_feat)
        scene_path_dict['pos'].append(pos_scene['scene_path'])
        scene_path_dict['pair_pos'].append(pair_pos_scene['scene_path'])
        scene_path_dict['neg'].append(neg_scene['scene_path'])

        # Load subclasses if they exist in scene_triplet
        if "obj_subclasses" in pos_scene.keys():
            scene_subclasses_dict['pos'].append(pos_scene['obj_subclasses'])
            scene_subclasses_dict['pair_pos'].append(pair_pos_scene['obj_subclasses'])
            scene_subclasses_dict['neg'].append(neg_scene['obj_subclasses'])
        if "obj_id" in pos_scene.keys():
            scene_obj_id_dict['pos'].append(pos_scene['obj_id'])
            scene_obj_id_dict['pair_pos'].append(pair_pos_scene['obj_id'])
            scene_obj_id_dict['neg'].append(neg_scene['obj_id'])

    pos_pcd = Pointclouds(pos_batch_dict['points'], features=pos_batch_dict['feats'])
    pair_pos_pcd = Pointclouds(pair_pos_batch_dict['points'], features=pair_pos_batch_dict['feats'])
    neg_pcd = Pointclouds(neg_batch_dict['points'], features=neg_batch_dict['feats'])
    pos_query_pcd = Pointclouds(pos_batch_dict['query_points'], features=pos_batch_dict['query_feats'])
    pair_pos_query_pcd = Pointclouds(pair_pos_batch_dict['query_points'], features=pair_pos_batch_dict['query_feats'])
    neg_query_pcd = Pointclouds(neg_batch_dict['query_points'], features=neg_batch_dict['query_feats'])

    scene_pcd = {'pos': pos_pcd, 'pair_pos': pair_pos_pcd, 'neg': neg_pcd, 'scene_path': scene_path_dict, 'scene_subclasses': scene_subclasses_dict, 'scene_obj_id': scene_obj_id_dict}
    query_pcd = {'pos': pos_query_pcd, 'pair_pos': pair_pos_query_pcd, 'neg': neg_query_pcd}

    return scene_pcd, query_pcd


def build_dense_3d_scene(scene_path, scene_name, remove_inst_list=None, obj_return_type='mesh', num_mesh_samples=10000, use_pbr=False, add_bbox=False, add_inst_name=False, add_sem_name=False, return_annot_scene=False):
    # Build scene by loading scene information
    # NOTE 1: instance labels in remove_inst_list are removed during scene building
    # NOTE 2: if obj_return_type == 'pcd', a list with Open3D geometry objects is returned
    # NOTE 3: if obj_return_type == 'mesh', a list of dictionaries containing geometry and material is returned
    # NOTE 4: If use_pbr == True, uv maps of triangles are reversed
    scene_triplet = np.load(scene_path)

    obj_scene_scales = scene_triplet[scene_name + "_obj_scene_scales"]
    obj_trans = scene_triplet[scene_name + "_trans"]
    obj_rot = scene_triplet[scene_name + "_rot"]
    scene_obj_paths = scene_triplet[scene_name + "_obj_path"]
    obj_list = []

    for obj_idx, scene_obj_path in enumerate(scene_obj_paths):
        if remove_inst_list is not None and obj_idx in remove_inst_list:
            continue
        texture_path = scene_obj_path.replace("raw_model.obj", "texture.png")
        if not os.path.exists(texture_path):
            texture_path = scene_obj_path.replace("raw_model.obj", "texture.jpg")
        tr_mesh = trimesh_load_with_postprocess(scene_obj_path, 'bottom_crop')
        tr_mesh.visual.material.image = Image.open(texture_path)
        tr_mesh.vertices *= obj_scene_scales[obj_idx: obj_idx + 1]
        tr_mesh.vertices[...] = \
            tr_mesh.vertices.dot(obj_rot[obj_idx].T) + obj_trans[obj_idx]

        if obj_return_type == 'mesh':
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(tr_mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(tr_mesh.faces)
            uvs = tr_mesh.visual.uv
            if not use_pbr:
                uvs[:, 1] = 1 - uvs[:, 1]
            triangles_uvs = []
            for i in range(3):
                triangles_uvs.append(uvs[tr_mesh.faces[:, i]].reshape(-1, 1, 2))
            triangles_uvs = np.concatenate(triangles_uvs, axis=1).reshape(-1, 2)

            o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
            texture_image = np.asarray(tr_mesh.visual.material.image)
            if texture_image.shape[-1] == 2:  # Greyscale textures should be converted to RGBA
                texture_image = np.stack([texture_image[..., 0]] * 3 + [texture_image[..., 1]], axis=-1)
            o3d_mesh.textures = [o3d.geometry.Image(texture_image)]
            o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(tr_mesh.faces))
            obj_list.append(o3d_mesh)
        else:  # Point cloud
            points, _, colors = sample_surface(tr_mesh, num_mesh_samples, sample_color=True)
            points = np.asarray(points)
            colors = np.asarray(colors)[:, :3] / 255.

            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(points)
            obj_pcd.colors = o3d.utility.Vector3dVector(colors)
            obj_list.append(obj_pcd)

    # NOTE: We assume floorplans are generated in a bi-level fashion
    floorplans = scene_triplet[scene_name + "_fp_points"]
    ground_level = floorplans[:, 1].min()
    floorplans = floorplans[:floorplans.shape[0] // 2, [0, 2]]  # Extract 2D contours
    floorplans = np.concatenate([floorplans[:, 0:1], ground_level * np.ones_like(floorplans[:, 0:1]), floorplans[:, 1:2]], axis=-1)  # Make to 3D
    floorplan_lines = np.stack([np.arange(floorplans.shape[0]), np.roll(np.arange(floorplans.shape[0]), shift=-1)], axis=1)

    floorplan_model = o3d.geometry.LineSet()
    floorplan_model.points = o3d.utility.Vector3dVector(floorplans)
    floorplan_model.lines = o3d.utility.Vector2iVector(floorplan_lines)
    dense_scene = obj_list + [floorplan_model]

    annot_scene = []
    if add_bbox:
        bbox_list = []
        for obj_idx, obj_model in enumerate(obj_list):
            bbox = o3d.geometry.AxisAlignedBoundingBox()
            if isinstance(obj_model, o3d.geometry.TriangleMesh):
                bbox = bbox.create_from_points(obj_model.vertices)
            else:  # Pointcloud
                bbox = bbox.create_from_points(obj_model.points)

            # Paint with random colors
            bbox.color = np.array([random.random(), random.random(), random.random()])
            bbox_list.append(bbox)
        dense_scene = dense_scene + bbox_list
        annot_scene = annot_scene + bbox_list

    if add_inst_name or add_sem_name:
        assert not (add_inst_name and add_sem_name)  # Both annotations cannot coexist
        text_list = []
        for obj_idx, obj_model in enumerate(obj_list):
            obj_cls = scene_triplet[scene_name + "_obj_classes_str"][obj_idx]
            if add_inst_name:
                text_model = o3d.t.geometry.TriangleMesh.create_text(f"{obj_idx}").to_legacy()
            if add_sem_name:
                text_model = o3d.t.geometry.TriangleMesh.create_text(f"{obj_cls}").to_legacy()
            flip_triangles = o3d.utility.Vector3iVector(
                np.asarray(text_model.triangles)[:, [2, 1, 0]]
            )
            full_triangles = o3d.utility.Vector3iVector(
                np.concatenate([
                    np.asarray(text_model.triangles),
                    flip_triangles
                ], axis=0)
            )
            text_model.triangles = full_triangles
            text_model = text_model.paint_uniform_color((1., 0., 0.))

            canonical_vertices = np.asarray(text_model.vertices)
            canonical_vertices = canonical_vertices - canonical_vertices.mean(axis=0, keepdims=True)
            text_scale_factor = 0.02
            transformed_vertices = canonical_vertices * obj_scene_scales[obj_idx].reshape(-1, 3) * text_scale_factor  # Sign added for flipping text
            align_rot = R.from_euler('x', -90, degrees=True).as_matrix()
            # NOTE: All text vertices are centered at zero
            transformed_vertices = transformed_vertices @ align_rot.T + obj_trans[obj_idx: obj_idx + 1]  # Omit rotation
            text_model.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            text_list.append(text_model)

        annot_scene = annot_scene + text_list

    if return_annot_scene:  # Optionally add annotated scene as output
        return dense_scene, annot_scene
    else:
        return dense_scene


def extract_query_points_from_inst(scene_path, inst_list: List[int], scene_name: str, num_points_per_obj: int = 1000, load_color: bool = False, point_sample_root: str = ""):
    scene_triplet = np.load(scene_path)

    obj_trans = scene_triplet[scene_name + "_trans"]
    obj_rot = scene_triplet[scene_name + "_rot"]
    obj_scene_scales = scene_triplet[scene_name + "_obj_scene_scales"]
    scene_obj_paths = scene_triplet[scene_name + "_obj_path"]

    query_points_list = []
    query_inst_labels_list = []
    query_points_colors_list = []

    for obj_idx, obj_path in enumerate(scene_obj_paths):
        if obj_idx not in inst_list:
            continue
        sample_path = os.path.join(
            point_sample_root,
            f"sample_{num_points_per_obj}",
            scene_triplet[scene_name + "_obj_id"][obj_idx] + ".npy"
        )
        if load_color or not os.path.exists(sample_path):
            obj_mesh = trimesh_load_with_postprocess(obj_path, 'bottom_crop')
            points, _, colors = sample_surface(obj_mesh, num_points_per_obj, sample_color=True)
            points = np.asarray(points)
            colors = np.asarray(colors)[:, :3] / 255.

            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(points)
            obj_pcd.colors = o3d.utility.Vector3dVector(colors)

            if len(obj_pcd.colors) == 0:
                obj_pcd.colors = o3d.utility.Vector3dVector(
                    np.ones_like(np.asarray(obj_pcd.points))
                )

            obj_pcd_np = np.asarray(obj_pcd.points)
            obj_rgb_np = np.asarray(obj_pcd.colors)

            # NOTE: Just for clarification, this is not cheating because we are essentially loading object points at the designated pose
            obj_pcd_np = obj_pcd_np * obj_scene_scales[obj_idx: obj_idx + 1]
            obj_pcd_np = obj_pcd_np @ obj_rot[obj_idx].T
            obj_pcd_np += obj_trans[obj_idx]

            query_points_list.append(obj_pcd_np)
            query_inst_labels_list.append(np.ones([obj_pcd_np.shape[0], ], dtype=int) * obj_idx)
            query_points_colors_list.append(obj_rgb_np)
        else:
            obj_pcd_np = np.load(sample_path)
            obj_pcd_np *= obj_scene_scales[obj_idx: obj_idx + 1]
            obj_pcd_np = obj_pcd_np @ obj_rot[obj_idx].T + obj_trans[obj_idx]

            query_points_list.append(obj_pcd_np)
            query_inst_labels_list.append(np.ones([obj_pcd_np.shape[0], ], dtype=int) * obj_idx)
            query_points_colors_list.append(-1 * np.ones(shape=[obj_pcd_np.shape[0], 3]))

    query_points = np.concatenate(query_points_list, axis=0)
    query_inst_labels = np.concatenate(query_inst_labels_list, axis=0)
    query_points_colors = np.concatenate(query_points_colors_list, axis=0)

    x_sort_idx = np.argsort(query_points[:, 0])
    y_sort_idx = np.argsort(query_points[x_sort_idx, 1])
    z_sort_idx = np.argsort(query_points[x_sort_idx[y_sort_idx], 2])
    query_points = query_points[x_sort_idx[y_sort_idx[z_sort_idx]]]
    query_inst_labels = query_inst_labels[x_sort_idx[y_sort_idx[z_sort_idx]]]
    query_points_colors = query_points_colors[x_sort_idx[y_sort_idx[z_sort_idx]]]
    query_inst_labels = query_inst_labels.reshape(-1, 1)

    query_feat = np.concatenate([query_inst_labels, query_points_colors], axis=-1)  # (N_pts, 1 + 3), representing 'instance' + 'colors'
    query_points, query_feat = torch.from_numpy(query_points).float(), torch.from_numpy(query_feat).float()
    query_pcd = Pointclouds(query_points[None, ...], features=query_feat[None, ...])

    return query_pcd


def get_histogram_scene_pairs(scene_path_list, num_classes):
    # Array containing semantic histograms
    full_semantic_hist = np.zeros([len(scene_path_list), num_classes])
    for path_idx, path in tqdm(enumerate(scene_path_list), desc="Extracting Histograms for Matching", total=len(scene_path_list)):
        scene_info = np.load(path)
        hist_vec = np.bincount(scene_info['pos_obj_classes'], minlength=num_classes).astype(float)  # / scene_info['pos_obj_classes'].shape[0]
        full_semantic_hist[path_idx] = hist_vec

    scene_pair_list = []
    for path_idx, path in tqdm(enumerate(scene_path_list), desc="Pair Generation", total=len(scene_path_list)):
        curr_hist_vec = full_semantic_hist[path_idx: path_idx + 1]
        hist_dist = np.linalg.norm(curr_hist_vec - full_semantic_hist, axis=-1)
        hist_dist[path_idx] = np.inf
        match_path_idx = hist_dist.argmin()
        scene_pair_list.append((path, scene_path_list[match_path_idx]))

    return scene_pair_list


def generate_obj_surface_points(scene_path_list: List[str], scene_name: str, num_samples_per_obj: int, point_sample_root: str = ""):
    points_list = []
    features_list = []
    for scene_path in scene_path_list:
        scene_triplet = np.load(scene_path)

        obj_scene_scales = scene_triplet[scene_name + "_obj_scene_scales"]
        obj_trans = scene_triplet[scene_name + "_trans"]
        obj_rot = scene_triplet[scene_name + "_rot"]
        scene_obj_paths = scene_triplet[scene_name + "_obj_path"]
        obj_list = []
        inst_labels_list = []
        colors_list = []

        for obj_idx, scene_obj_path in enumerate(scene_obj_paths):
            sample_path = os.path.join(
                point_sample_root,
                f"sample_{num_samples_per_obj}",
                scene_triplet[scene_name + "_obj_id"][obj_idx] + ".npy"
            )
            if os.path.exists(sample_path):  # Use pre-sampled points
                scene_sample_pcd_np = np.load(sample_path)
                scene_sample_pcd_np *= obj_scene_scales[obj_idx: obj_idx + 1]
                scene_sample_pcd_np = scene_sample_pcd_np @ obj_rot[obj_idx].T + obj_trans[obj_idx]

                points = torch.from_numpy(scene_sample_pcd_np).float()
                obj_list.append(points)
                inst_labels_list.append(torch.ones([points.shape[0], 1], dtype=int) * obj_idx)
                colors_list.append(torch.ones([points.shape[0], 3], dtype=float) * -1)
            else:
                tr_mesh = trimesh_load_with_postprocess(scene_obj_path, 'bottom_crop')
                tr_mesh.vertices *= obj_scene_scales[obj_idx: obj_idx + 1]
                tr_mesh.vertices[...] = \
                    tr_mesh.vertices.dot(obj_rot[obj_idx].T) + obj_trans[obj_idx]

                points, _ = sample_surface(tr_mesh, num_samples_per_obj, sample_color=False)
                points = np.asarray(points)

                points = torch.from_numpy(points).float()
                obj_list.append(points)
                inst_labels_list.append(torch.ones([points.shape[0], 1], dtype=int) * obj_idx)
                colors_list.append(torch.ones([points.shape[0], 3], dtype=float) * -1)

        points = torch.cat(obj_list, dim=0)
        points_list.append(points)

        feats = torch.cat([torch.cat([inst_labels, colors], dim=-1) for inst_labels, colors in zip(inst_labels_list, colors_list)], dim=0)
        features_list.append(feats)
    surface_pcd = Pointclouds(points=points_list, features=features_list)

    return surface_pcd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Debug configs
    parser.add_argument("--vis_tgt", help="Target to visualize for debugging", default="scene_pts", type=str)

    # General configs
    parser.add_argument("--log_dir", help="Log directory for saving experiment results", default="./log/")
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)

    # Dataset configs
    parser.add_argument("--point_feat_extractor", help="Type of point feature extractor to use for dataset generation", default=None, type=str)
    parser.add_argument("--feat_sample_points", type=int, help="Number of points to sample per object model for feature extraction", default=2048)
    parser.add_argument("--scene_sample_points", type=int, help="Number of points to sample per object model for scene generation", default=50)
    parser.add_argument("--obj_json", help=".json file containing information on 3D-FUTURE meshes", default="./data/3D-FUTURE-model/model_info.json")
    parser.add_argument("--feat_pca_root", help="Root folder containing PCA components for point feature dimension reduction (currently used for Vector Neurons)", default="./data/3d_future_pca/")
    parser.add_argument("--point_sample_root", help="Root folder containing point samples from object mesh files (currently used for scene generation without feature extraction)", default="./data/3d_future_point_samples/")
    parser.add_argument("--random_group_topk_range", type=int, help="Minimum and maximum value for top-K nearest object sampling during random group query sampling", default=[1, 4], nargs=2)

    # Training configs
    parser.add_argument("--scene_root", help="Root directory containing scene data", type=str, default='./data/3d_front_scenes/3d_front_large_noise/')
    parser.add_argument("--batch_size", help="Batch size to use for training", default=8, type=int)
    parser.add_argument("--epochs", help="Number of epochs to run for training", default=10000, type=int)
    parser.add_argument("--mode", help="Mode to use training (either train or test)", default="train", type=str)
    parser.add_argument("--learning_rate", help="Learning rate for training feature field", default=0.0001, type=float)
    parser.add_argument("--loss_type", help="Type of loss to use for training", default="infonce_paired", type=str)
    parser.add_argument("--triplet_margin", help="Margin value for triple loss", default=0.5, type=float)
    parser.add_argument("--infonce_temp", help="InfoNCE temperature parameter", default=0.2, type=float)
    parser.add_argument("--scene_buffer_size", help="Size of buffer to use for loading scenes each epoch", default=32, type=int)
    parser.add_argument("--query_point_type", help="Type of query point sampling to use for generating trianing samples", default="obj_points", type=str)
    parser.add_argument("--num_vert_split", help="Number of vertical splits to make for floorplan queries", default=1, type=int)
    parser.add_argument("--force_up_margin", help="Forced upper margin for floorplan queries (defaults to using maximum height of walls)", default=0.5, type=float)
    parser.add_argument("--force_low_margin", help="Forced lower margin for floorplan queries (defaults to using minimum height of walls)", default=0.5, type=float)
    parser.add_argument("--num_cache_query", help="Number of query point locations to cacher per scene", default=50000, type=int)
    parser.add_argument("--num_sample_query", help="Number of query points to sample per scene", default=32, type=int)
    parser.add_argument("--update_buffer_every", help="Number of epochs before updating buffer", default=128, type=int)
    parser.add_argument("--fp_feat_type", help="Type of floorplan features to use", default="learned", type=str)
    parser.add_argument("--obj_point_query_scale_factor", help="Scale factor to use for object point query sampling", default=None, type=float, nargs="+")
    parser.add_argument("--query_sampling_method", help="Type of query sampling to use for training deformation field", default="region", type=str)
    parser.add_argument("--fp_point_type", help="Type of floorplan points to use", default="wireframe", type=str)
    parser.add_argument("--fp_sample_step_size", help="Step size (contour, height) for generating floorplan sampled points", default=[0.3, 0.3], type=float, nargs=2)
    parser.add_argument("--fp_label_type", help="Type of floorplan labeling to use", default="single", type=str)

    args = parser.parse_args()

    # Fix seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_scene_path_list = glob(os.path.join(args.scene_root, '**/*.npz'))
    test_scene_path_list = full_scene_path_list[:args.scene_buffer_size]

    keypoint_buffer = KeypointSceneBuffer(
        scene_path_list=test_scene_path_list,
        cfg=args,
        device=device
    )
    data_loader = DataLoader(
        keypoint_buffer,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=keybuff_collate_fn
    )

    for batch_idx, (scene_pcd, query_pcd) in enumerate(data_loader):
        print(f"Batch {batch_idx}")
        print(f"pos: scene {len(scene_pcd['pos'])} scenes, query {len(query_pcd['pos'])} scenes")
        print(f"neg: scene {len(scene_pcd['neg'])} scenes, query {len(query_pcd['neg'])} scenes")
        print(f"pair_pos: scene {len(scene_pcd['pair_pos'])} scenes, query {len(query_pcd['pair_pos'])} scenes")

        # Visualization of query points
        for s_idx in range(len(scene_pcd['pos'])):
            dense_scenes = {}
            dense_scenes['pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pos'][s_idx], 'pos')
            dense_scenes['pair_pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pair_pos'][s_idx], 'pair_pos')
            dense_scenes['neg'] = build_dense_3d_scene(scene_pcd['scene_path']['neg'][s_idx], 'neg')

            print(scene_pcd['scene_path'][s_idx])
            for scene_name in ['pos', 'neg', 'pair_pos']:
                if args.vis_tgt == "scene_pts":
                    scene_points = o3d.geometry.PointCloud()
                    scene_points.points = o3d.utility.Vector3dVector(scene_pcd[scene_name].points_list()[s_idx].cpu().numpy())
                    o3d.visualization.draw_geometries(dense_scenes[scene_name] + [scene_points])
                elif args.vis_tgt == "query_pts":
                    query_points = o3d.geometry.PointCloud()
                    query_points.points = o3d.utility.Vector3dVector(query_pcd[scene_name].points_list()[s_idx].cpu().numpy())
                    o3d.visualization.draw_geometries(dense_scenes[scene_name] + [query_points])
                else:
                    raise NotImplementedError("Other visualization targets not supported")

        # Test k-NN from query points to point cloud
        pos_knn = knn_points(
            query_pcd['pos'].points_padded(),
            scene_pcd['pos'].points_padded(),
            lengths1=query_pcd['pos'].num_points_per_cloud(),
            lengths2=scene_pcd['pos'].num_points_per_cloud(),
            K=30
        )
        pos_knn_dists = pos_knn.dists  # (N_batch, N_query, K)
        pos_knn_idx = pos_knn.idx  # (N_batch, N_query, K)
        pos_knn_feats = knn_gather(
            scene_pcd['pos'].features_padded(),
            pos_knn_idx,
            scene_pcd['pos'].num_points_per_cloud()
        )  # (N_batch, N_query, K, 1 + 1 + D_emb)

        # Test ball query from query points to point cloud
        pos_ball = ball_query(
            query_pcd['pos'].points_padded(),
            scene_pcd['pos'].points_padded(),
            lengths1=query_pcd['pos'].num_points_per_cloud(),
            lengths2=scene_pcd['pos'].num_points_per_cloud(),
            K=300,
            radius=1.,
            return_nn=True
        )
        pos_ball_dists = pos_ball.dists  # (N_batch, N_query, K)
        pos_ball_idx = pos_ball.idx  # (N_batch, N_query, K)
        pos_ball_feats = masked_gather(scene_pcd['pos'].features_padded(), pos_ball_idx)
