import argparse
import torch
import numpy as np
import random
import os
from glob import glob
from arkit.utils import (
    RollingSampler,
    print_state,
    generate_random_region_2d,
    choice_without_replacement,
    get_color_wheel,
    map_coordinates_to_color,
    keypoints_to_spheres,
    o3d_geometry_list_shift,
    o3d_geometry_list_scale,
    o3d_geometry_list_aabb
)
from arkit.eval_utils import (
    map_accuracy,
    chamfer_accuracy,
    MetricLogger,
    symmetry_aware_point_accuracy
)
from arkit.data_utils import (
    KeypointSceneBuffer,
    keybuff_collate_fn,
    build_dense_3d_scene,
    extract_query_points_from_inst,
    get_histogram_scene_pairs,
    generate_obj_surface_points
)
from torch.utils.data import DataLoader
import open3d as o3d
from collections import namedtuple
import cv2
from scipy.spatial.transform import Rotation as R
from arkit.global_field_match import AffineMatcher
from arkit.local_field_match import PointMatcher
from matplotlib import colormaps
import wandb
from tqdm import tqdm
import pandas as pd
import triangle as tr
import trimesh
from PIL import Image

# Ideal scene bounding box size for visualization
IDEAL_VIS_LENGTH = np.array([12.0, 3.2, 12.0])


class RelMatchEvaluator():
    def __init__(self, cfg, log_dir):
        # NOTE: We aim to match a group of objects in 'pair_pos' to those in 'pos'
        self.cfg = cfg
        self.mode = cfg.mode
        self.batch_size = cfg.batch_size
        self.log_dir = log_dir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Setup local feature field
        assert self.cfg.load_local_feature_field is not None
        print(f"Loading model from {self.cfg.load_local_feature_field}")
        self.local_feature_field = torch.load(self.cfg.load_local_feature_field).to(self.device)

        # Freeze parameters for local feature field
        self.local_feature_field = self.local_feature_field.eval()
        for param in self.local_feature_field.parameters():
            param.requires_grad = False
        self.local_feature_field.to(self.device)

        # Setup global feature field
        assert self.cfg.load_global_feature_field is not None
        print(f"Loading model from {self.cfg.load_global_feature_field}")
        self.global_feature_field = torch.load(self.cfg.load_global_feature_field).to(self.device)

        # Freeze parameters for local feature field
        self.global_feature_field = self.global_feature_field.eval()
        for param in self.global_feature_field.parameters():
            param.requires_grad = False
        self.global_feature_field.to(self.device)

        # Setup configs for scene buffer
        self.scene_buffer_size = cfg.scene_buffer_size
        self.scene_root = cfg.scene_root
        self.query_point_type = cfg.query_point_type
        self.full_scene_path_list = sorted(glob(os.path.join(self.scene_root, '**/*.npz')))

        # NOTE 1: Here we list up pairs of paths (pos_path & pair_pos_path) which are kept identical for training
        # NOTE 2: pos_path is used for loading scene information for pos / neg samples and pair_pos_path is used for loading scene information for pair_pos samples
        if self.cfg.scene_pair_type == "identical":
            self.test_scene_path_pair_list = [(path, path) for path in self.full_scene_path_list if 'test' in path]
        elif self.cfg.scene_pair_type == "augment":
            if cfg.scene_augment_root is not None:
                scene_augment_path_list = sorted(glob(os.path.join(cfg.scene_augment_root, '**/*.npz')))
            else:
                scene_augment_root = self.scene_root.strip('/') + '_augment/'
                scene_augment_path_list = [path.replace(self.scene_root, scene_augment_root) for path in self.full_scene_path_list]
            self.test_scene_path_pair_list = [(path, aug_path) for path, aug_path in zip(self.full_scene_path_list, scene_augment_path_list) if 'test' in path]
        elif self.cfg.scene_pair_type == "random":
            test_scene_path_list = [path for path in self.full_scene_path_list if 'test' in path]
            self.test_scene_path_pair_list = [(path, random.choice(test_scene_path_list)) for path in self.full_scene_path_list if 'test' in path]
        elif self.cfg.scene_pair_type == "hist_match":
            test_scene_path_list = [path for path in self.full_scene_path_list if 'test' in path]
            self.test_scene_path_pair_list = get_histogram_scene_pairs(test_scene_path_list, self.cfg.num_classes)
        elif self.cfg.scene_pair_type == "manual":
            assert self.cfg.scene_pair_file is not None
            scene_pair_table = pd.read_table(self.cfg.scene_pair_file)
            ref_scene_path_list = [os.path.join(self.cfg.scene_root, path, "scene.npz") for path in scene_pair_table.ref]
            tgt_scene_path_list = [os.path.join(self.cfg.scene_root, path, "scene.npz") for path in scene_pair_table.tgt]
            self.test_scene_path_pair_list = [(ref_path, tgt_path) for (ref_path, tgt_path) in zip(ref_scene_path_list, tgt_scene_path_list)]
        else:
            raise NotImplementedError("Other scene pair types not supported")
        self.test_scene_path_pair_sampler = RollingSampler(self.test_scene_path_pair_list)

        if self.scene_buffer_size == -1:  # Set to the number of scene path pairs if scene_buffer_size is -1
            self.scene_buffer_size = len(self.test_scene_path_pair_list)

        if self.cfg.scene_pair_file is None:  # Treat all scene pairs to be valid
            self.valid_pairs = self.test_scene_path_pair_list + [(pair_1, pair_0) for (pair_0, pair_1) in self.test_scene_path_pair_list]  # Order-agnostic scene pairs
            self.valid_pairs = [(os.path.normpath(pair_0), os.path.normpath(pair_1)) for (pair_0, pair_1) in self.valid_pairs]
        else:  # Only treat manually specified scene pairs to be valid
            assert self.cfg.scene_pair_type == "manual"
            self.valid_pairs = [(os.path.normpath(pair_0), os.path.normpath(pair_1)) for path_idx, (pair_0, pair_1) in enumerate(self.test_scene_path_pair_list) \
                if isinstance(scene_pair_table.ref_inst[path_idx], str)]

        cfg_dict = {k: v for k, v in vars(self.cfg).items()}
        input_override_dict = {
            'query_point_type': self.query_point_type,
            'query_sampling_method': 'random_group',
            'num_cache_query': self.cfg.num_query,  # Number of points to sample per object for fine warping
            'num_sample_query': self.cfg.num_query,  # Equally set
            'enforce_sampling': True
        }
        Config = namedtuple('Config', tuple(set(tuple(cfg_dict) + tuple(input_override_dict.keys()))))
        cfg_dict.update(input_override_dict)
        self.input_cfg = Config(**cfg_dict)  # Config used for generating matching point inputs

        if self.cfg.global_matcher_type == 'affine':
            self.global_matcher = AffineMatcher(self.cfg, self.device, self.global_feature_field)
        else:
            raise NotImplementedError("Other global matchers not supported")

        if self.cfg.local_matcher_type == 'point':
            self.local_matcher = PointMatcher(self.cfg, self.device, self.local_feature_field)
        else:
            raise NotImplementedError("Other local matchers not supported")

        # Initialize metric logger for evaluation
        self.metric_list = [f"map_acc_{thres}" for thres in self.cfg.map_acc_thres] + \
            [f"bijectivity_acc_{thres}" for thres in self.cfg.bijectivity_acc_thres] + \
            [f"chamfer_acc_{thres}" for thres in self.cfg.chamfer_acc_thres]
        self.metric_logger = MetricLogger(self.metric_list)

        # Initialize transform dictionary containing transforms found from tested scene pairs
        self.estim_transform_dict = {}

        # Make directory for optionally saving scene meshes
        if args.save_scene_mesh or args.save_transfer:
            self.mesh_save_dir = os.path.join(args.log_dir, "scene_meshes")
            if not os.path.exists(self.mesh_save_dir):
                os.makedirs(self.mesh_save_dir, exist_ok=True)
            self.fp_texture_path = args.fp_texture_path

    def reset_scene_buffer(self):
        scene_path_pair_list = self.test_scene_path_pair_sampler.sample(self.scene_buffer_size)

        self.test_scene_buffer = KeypointSceneBuffer(
            scene_path_pair_list=scene_path_pair_list,
            cfg=self.input_cfg,
            device=self.device
        )
        self.test_loader = DataLoader(
            self.test_scene_buffer,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=keybuff_collate_fn
        )

    def run(self):
        for eval_idx in range(self.cfg.eval_reps):
            self.reset_scene_buffer()

            if self.mode == "regular":
                self.eval(eval_idx)
            else:
                raise NotImplementedError("Other evaluation modes not supported")

        # Save metric logger
        metric_save_path = os.path.join(self.log_dir, f"{self.cfg.scene_pair_type}_metrics.pkl")
        self.metric_logger.save_to_path(metric_save_path)

        # Save estimated transforms
        transform_save_path = os.path.join(self.log_dir, f"{self.cfg.scene_pair_type}_transforms.pth")
        torch.save(self.estim_transform_dict, transform_save_path)

    def eval(self, epoch):
        for batch_idx, (scene_pcd, query_pcd) in enumerate(self.test_loader):
            num_batch_scenes = len(scene_pcd['pos'])  # For each triplet and local / global points, number of scenes is fixed
            # Prepare query points for affine matchment
            pair_pos_query = query_pcd['pair_pos'].to(self.device)  # Set as random group points
            pos_query = generate_obj_surface_points(scene_pcd['scene_path']['pos'], 'pos',
                self.cfg.num_query, self.cfg.point_sample_root).to(self.device)  # All points from object surfaces are considered

            # List up instances to perform global / local matchment
            global_transform_list, global_inst_match_list = self.global_matcher.find_transforms(pair_pos_query, pos_query, scene_pcd['pair_pos'].to(self.device), scene_pcd['pos'].to(self.device))
            # Estimate local transform
            if self.cfg.skip_local_matching:
                assert self.cfg.vis_local_match_mode is None
                local_transform_list = [[] for _ in range(num_batch_scenes)]
            else:
                local_transform_list, local_inst_match_list = self.local_matcher.find_transforms(global_transform_list, global_inst_match_list, pair_pos_query, pos_query, scene_pcd['pair_pos'].to(self.device), scene_pcd['pos'].to(self.device))

            # Determine of matches exist from input data
            match_exist_list = []
            num_points_list = []
            for scene_idx in range(num_batch_scenes):
                match_exist = (os.path.normpath(scene_pcd['scene_path']['pos'][scene_idx]), os.path.normpath(scene_pcd['scene_path']['pair_pos'][scene_idx])) in self.valid_pairs
                num_points = query_pcd['pair_pos'].points_list()[scene_idx].shape[0]
                match_exist_list.append(match_exist)
                num_points_list.append(num_points)

            # Optionally print metrics against pseudo ground-truth warps (applicable when query points are obtained with obj_match_points)
            if self.cfg.query_point_type == "obj_match_points":
                for scene_idx in range(num_batch_scenes):
                    if match_exist_list[scene_idx]:  # Match exists, then measure ratio of points within error threshold
                        if len(local_transform_list[scene_idx]) != 0:
                            acc_list = map_accuracy(
                                transform=local_transform_list[scene_idx][0],  # NOTE: We only evaluate the best match found
                                eval_points=query_pcd['pair_pos'].points_list()[scene_idx].cpu(),
                                gt_points=query_pcd['pos'].points_list()[scene_idx].cpu(),
                                thres_list=self.cfg.map_acc_thres
                            )
                        else:
                            acc_list = [0. for _ in self.cfg.map_acc_thres]
                    else:
                        if len(local_transform_list[scene_idx]) == 0:
                            acc_list = [1. for _ in self.cfg.map_acc_thres]
                        else:
                            acc_list = [0. for _ in self.cfg.map_acc_thres]
                    # Update metrics
                    for thres_idx, thres in enumerate(self.cfg.map_acc_thres):
                        self.metric_logger.update_values(
                            new_metric={f"map_acc_{thres}": acc_list[thres_idx]},
                            num_data_points={f"map_acc_{thres}": num_points_list[scene_idx]}
                        )
            else:
                for scene_idx in range(num_batch_scenes):
                    acc_list = [0. for _ in self.cfg.map_acc_thres]
                # Update metrics
                for thres_idx, thres in enumerate(self.cfg.map_acc_thres):
                    self.metric_logger.update_values(
                        new_metric={f"map_acc_{thres}": acc_list[thres_idx]},
                        num_data_points={f"map_acc_{thres}": num_points_list[scene_idx]}
                    )

            # Chamfer distance loss
            for scene_idx in range(num_batch_scenes):
                if match_exist_list[scene_idx]:  # Match exists, then measure ratio of groups within error threshold
                    if len(local_transform_list[scene_idx]) != 0:
                        acc_list = chamfer_accuracy(
                            transform=local_transform_list[scene_idx][0],  # NOTE: We only evaluate the best match found
                            eval_points=pair_pos_query.points_list()[scene_idx].cpu(),
                            gt_points=pos_query.points_list()[scene_idx].cpu(),
                            eval_inst_labels=pair_pos_query.features_list()[scene_idx][:, 0].cpu(),
                            gt_inst_labels=pos_query.features_list()[scene_idx][:, 0].cpu(),
                            thres_list=self.cfg.chamfer_acc_thres
                        )
                    else:
                        acc_list = [0. for _ in self.cfg.chamfer_acc_thres]
                else:
                    if len(local_transform_list[scene_idx]) == 0:
                        acc_list = [1. for _ in self.cfg.chamfer_acc_thres]
                    else:
                        acc_list = [0. for _ in self.cfg.chamfer_acc_thres]

                # Update metrics
                for thres_idx, thres in enumerate(self.cfg.chamfer_acc_thres):
                    self.metric_logger.update_values(
                        new_metric={f"chamfer_acc_{thres}": acc_list[thres_idx]},
                        num_data_points={f"chamfer_acc_{thres}": 1}
                    )

            # Measure bijectivity
            inv_local_transform_list = [[] for _ in range(num_batch_scenes)]
            for scene_idx in range(num_batch_scenes):
                if len(local_transform_list[scene_idx]) != 0:
                    # Prepare query points for affine matchment
                    # NOTE: For bijectivity evaluation we only consider the first estimated transform with the best cost
                    inv_pair_pos_query = extract_query_points_from_inst(scene_pcd['scene_path']['pos'][scene_idx], inst_list=local_inst_match_list[scene_idx][0].long().tolist(),
                        scene_name='pos', num_points_per_obj=self.cfg.num_query, point_sample_root=self.cfg.point_sample_root).to(self.device)  # NOTE: Here we take 'pos' scene points as we are estimating the inverse
                    inv_pos_query = generate_obj_surface_points(scene_pcd['scene_path']['pair_pos'][scene_idx: scene_idx + 1], 'pair_pos',
                        self.cfg.num_query, self.cfg.point_sample_root).to(self.device)  # All points from object surfaces are considered (NOTE: Here we take 'pair_pos' scene points as we are estimating the inverse)
                    # NOTE: Why are 'pair_pos' and 'pos' directly used unlike build_dense_3d_scenes below? Because query and scene points are alreadly correctly sampled in data_utilspy

                    # List up instances to perform global / local matchment (NOTE: 'pos' and 'pair_pos' scenes are inversed)
                    inv_global_transform_list, inv_global_inst_match_list = self.global_matcher.find_transforms(inv_pair_pos_query, inv_pos_query, scene_pcd['pos'][scene_idx].to(self.device), scene_pcd['pair_pos'][scene_idx].to(self.device))

                    # Estimate local transform (NOTE: 'pos' and 'pair_pos' scenes are inversed)
                    inv_local_transform, inv_local_inst_match = self.local_matcher.find_transforms(inv_global_transform_list, inv_global_inst_match_list, inv_pair_pos_query, inv_pos_query, scene_pcd['pos'][scene_idx].to(self.device), scene_pcd['pair_pos'][scene_idx].to(self.device))

                    # Cache inverse transforms
                    inv_local_transform_list[scene_idx] = inv_local_transform[0]

                    if match_exist_list[scene_idx]:  # Match exists, then measure ratio of points within error threshold
                        if len(inv_local_transform[0]) != 0:
                            transform = local_transform_list[scene_idx][0]
                            inv_transform = inv_local_transform[0][0]  # We consider the first transform for bijectivity evaluation
                            pair_pos_query_coords = pair_pos_query.points_list()[scene_idx].cpu()
                            acc_list = symmetry_aware_point_accuracy(
                                eval_points=inv_transform(transform(pair_pos_query_coords)), 
                                gt_points=pair_pos_query_coords,
                                thres_list=self.cfg.bijectivity_acc_thres
                            )
                        else:
                            acc_list = [0. for _ in self.cfg.bijectivity_acc_thres]
                    else:
                        if len(inv_local_transform[0]) == 0:
                            acc_list = [1. for _ in self.cfg.bijectivity_acc_thres]
                        else:
                            acc_list = [0. for _ in self.cfg.bijectivity_acc_thres]
                    # Update metrics
                    for thres_idx, thres in enumerate(self.cfg.bijectivity_acc_thres):
                        self.metric_logger.update_values(
                            new_metric={f"bijectivity_acc_{thres}": acc_list[thres_idx]},
                            num_data_points={f"bijectivity_acc_{thres}": num_points_list[scene_idx]}
                        )
                else:
                    if match_exist_list[scene_idx]:
                        acc_list = [0. for _ in self.cfg.bijectivity_acc_thres]
                    else:
                        acc_list = [1. for _ in self.cfg.bijectivity_acc_thres]
                    # Update metrics
                    for thres_idx, thres in enumerate(self.cfg.bijectivity_acc_thres):
                        self.metric_logger.update_values(
                            new_metric={f"bijectivity_acc_{thres}": acc_list[thres_idx]},
                            num_data_points={f"bijectivity_acc_{thres}": num_points_list[scene_idx]}
                        )

            print_dict = {
                'Iter': batch_idx,
                **self.metric_logger.metric_avg,
                **self.metric_logger.metric_latest
            }

            if self.cfg.wandb:
                wandb.log(
                    dict(**self.metric_logger.metric_avg, **self.metric_logger.metric_latest)
                )

            print_state(print_dict)

            # Cache estimated transforms
            for scene_idx in range(num_batch_scenes):
                tgt_scene_path = scene_pcd['scene_path']['pair_pos'][scene_idx]
                ref_scene_path = scene_pcd['scene_path']['pos'][scene_idx]

                scene_pair_key = (os.path.normpath(ref_scene_path), os.path.normpath(tgt_scene_path))
                self.estim_transform_dict[scene_pair_key] = {}
                self.estim_transform_dict[scene_pair_key]['transforms'] = [
                    transform.cpu() for transform in local_transform_list[scene_idx]
                ]
                self.estim_transform_dict[scene_pair_key]['tgt_inst_labels'] = \
                    query_pcd['pair_pos'].features_list()[scene_idx][:, 0].long().unique().tolist()  # Instance labels of input objects for matching
                self.estim_transform_dict[scene_pair_key]['inv_transforms'] = [
                    transform.cpu() for transform in inv_local_transform_list[scene_idx]
                ]

                # Manually change device of local transforms for saving
                for t_idx in range(len(self.estim_transform_dict[scene_pair_key]['transforms'])):
                    self.estim_transform_dict[scene_pair_key]['transforms'][t_idx].device = torch.device('cpu')
                for t_idx in range(len(self.estim_transform_dict[scene_pair_key]['inv_transforms'])):
                    self.estim_transform_dict[scene_pair_key]['inv_transforms'][t_idx].device = torch.device('cpu')

            # Optionally save input meshes
            if self.cfg.save_scene_mesh:
                for scene_idx in range(num_batch_scenes):
                    dense_scenes = {}

                    dense_scenes['pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pos'][scene_idx], 'pos', remove_obj_mesh=remove_obj_mesh_from_full_scene)
                    dense_scenes['pair_pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pair_pos'][scene_idx], vis_pair_scene_name, remove_obj_mesh=remove_obj_mesh_from_full_scene)

                    for scene_name in ['pos', 'pair_pos']:
                        mesh_root = os.path.join(self.mesh_save_dir, f"scene_{epoch}_{batch_idx}_{scene_idx}_{scene_name}")

                        if not os.path.exists(mesh_root):
                            os.makedirs(mesh_root, exist_ok=True)

                        obj_count = 0
                        for scene_entity in tqdm(dense_scenes[scene_name], desc=f"Saving {scene_name} scene {scene_idx}"):
                            # Add meshes and linesets to scene_mesh
                            if isinstance(scene_entity, o3d.geometry.TriangleMesh):
                                mesh_path = os.path.join(mesh_root, f"obj_{obj_count}.obj")  # .obj used to save texture
                                o3d.io.write_triangle_mesh(mesh_path, scene_entity)
                                obj_count += 1
                            elif isinstance(scene_entity, o3d.geometry.LineSet):
                                fp_path = os.path.join(mesh_root, "fp.obj")  # .obj used to save color and texture
                                lines = np.array(scene_entity.lines)
                                vertices = np.array(scene_entity.points)

                                triangulated = tr.triangulate({'vertices': vertices[:,[0, 2]], 'segments': lines}, 'p')
                                triangles = triangulated['triangles']
                                vertices = np.copy(triangulated['vertices']) # new vertices are added if given vertices are not enough
                                vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=1)[:, [0,2,1]]

                                fp_model = o3d.geometry.TriangleMesh()
                                fp_model.vertices = o3d.utility.Vector3dVector(vertices)
                                fp_model.triangles = o3d.utility.Vector3iVector(triangles)
                                fp_model.paint_uniform_color([0.8, 0.8, 0.8])

                                uv = np.copy(vertices[:, [0, 2]])
                                uv -= uv.min(axis=0)
                                uv /= 1.0  # repeat every 1m
                                texture_path = self.fp_texture_path

                                tr_floor = trimesh.Trimesh(
                                    np.copy(vertices), np.copy(triangles), process=False
                                )
                                tr_floor.visual = trimesh.visual.TextureVisuals(
                                    uv=np.copy(uv),
                                    material=trimesh.visual.material.SimpleMaterial(
                                        image=Image.open(texture_path)
                                    )
                                )
                                uvs = tr_floor.visual.uv
                                triangles_uvs = []
                                for i in range(3):
                                    triangles_uvs.append(uvs[tr_floor.faces[:, i]].reshape(-1, 1, 2))
                                triangles_uvs = np.concatenate(triangles_uvs, axis=1).reshape(-1, 2)

                                fp_model.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
                                fp_model.textures = [o3d.geometry.Image(np.asarray(tr_floor.visual.material.image))]
                                fp_model.triangle_material_ids = o3d.utility.IntVector([0] * len(tr_floor.faces))

                                o3d.io.write_triangle_mesh(fp_path, fp_model)

            # Optionally visualize input scenes

            """
                NOTE: For real heterogeneous scenes (from hist_match & random scene_pair_type), we use the 'pos' scene from the 'pair_pos' scene path.
                Namely, this is the scene path retrieved from histogram matching.
                Such a choice is made since 'pos' scenes are collected from real scans while 'pair_pos' scenes are generated synthetically.
                This choice is specific to ARKiT scenes.
            """

            if self.cfg.scene_pair_type in ["random", "hist_match"]:  # Real heterogenous scenes
                vis_pair_scene_name = 'pos'
                remove_obj_mesh_from_full_scene = False
            else:
                vis_pair_scene_name = 'pair_pos'
                remove_obj_mesh_from_full_scene = True

            if self.cfg.vis_input_mode is not None:
                num_batches = self.scene_buffer_size // self.cfg.batch_size + 1 if self.scene_buffer_size % self.cfg.batch_size != 0 else self.scene_buffer_size // self.cfg.batch_size
                for scene_idx in range(num_batch_scenes):
                    vis_sample_idx = num_batches * num_batch_scenes * epoch + batch_idx * num_batch_scenes + scene_idx

                    # Visualize query both points if scene pairs are homogenous or synthetically augmented
                    vis_query_pair_pos = query_pcd['pair_pos'].points_list()[scene_idx].cpu()

                    if self.cfg.scene_pair_type in ['augment', 'identical']:
                        vis_query_pos = query_pcd['pos'].points_list()[scene_idx].cpu()
                    else:
                        vis_query_pos = None
                    num_query = vis_query_pair_pos.shape[0]
                    idx_color = np.linspace(0., 1., num_query)  # (N_query, )
                    idx_color = colormaps['jet'](idx_color, alpha=False, bytes=False)[:, :3]

                    if self.cfg.vis_sort_query == 'xyz':
                        x_sort_idx = np.argsort(vis_query_pair_pos[:, 0])
                        y_sort_idx = np.argsort(vis_query_pair_pos[x_sort_idx, 1])
                        z_sort_idx = np.argsort(vis_query_pair_pos[x_sort_idx[y_sort_idx], 2])
                        vis_query_pair_pos = vis_query_pair_pos[x_sort_idx[y_sort_idx[z_sort_idx]]]
                        if vis_query_pos is not None:
                            vis_query_pos = vis_query_pos[x_sort_idx[y_sort_idx[z_sort_idx]]]
                    elif self.cfg.vis_sort_query == 'yzx':
                        y_sort_idx = np.argsort(vis_query_pair_pos[:, 1])
                        z_sort_idx = np.argsort(vis_query_pair_pos[y_sort_idx, 2])
                        x_sort_idx = np.argsort(vis_query_pair_pos[y_sort_idx[z_sort_idx], 0])
                        vis_query_pair_pos = vis_query_pair_pos[y_sort_idx[z_sort_idx[x_sort_idx]]]
                        if vis_query_pos is not None:
                            vis_query_pos = vis_query_pos[y_sort_idx[z_sort_idx[x_sort_idx]]]
                    elif self.cfg.vis_sort_query == 'zxy':
                        z_sort_idx = np.argsort(vis_query_pair_pos[:, 2])
                        x_sort_idx = np.argsort(vis_query_pair_pos[z_sort_idx, 0])
                        y_sort_idx = np.argsort(vis_query_pair_pos[z_sort_idx[x_sort_idx], 1])
                        vis_query_pair_pos = vis_query_pair_pos[z_sort_idx[x_sort_idx[y_sort_idx]]]
                        if vis_query_pos is not None:
                            vis_query_pos = vis_query_pos[z_sort_idx[x_sort_idx[y_sort_idx]]]
                    else:
                        raise NotImplementedError("Other sorting methods not supported")

                    dense_scenes = {}
                    dense_scenes['pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pos'][scene_idx], 'pos', remove_obj_mesh=remove_obj_mesh_from_full_scene)
                    dense_scenes['pair_pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pair_pos'][scene_idx], vis_pair_scene_name, remove_obj_mesh=remove_obj_mesh_from_full_scene)
                    dense_scenes['neg'] = build_dense_3d_scene(scene_pcd['scene_path']['neg'][scene_idx], 'neg', remove_obj_mesh=remove_obj_mesh_from_full_scene)

                    if self.cfg.vis_input_mode == 'scene':
                        vis_tgt_scene = dense_scenes['pos']
                        vis_ref_scene = dense_scenes['pair_pos']
                    elif self.cfg.vis_input_mode == 'input':
                        # Prepare positive query points
                        vis_pair_pos = np.concatenate([vis_query_pair_pos, idx_color], axis=-1)  # (N_query, 6)
                        vis_pair_pos_pcd = o3d.geometry.PointCloud()
                        vis_pair_pos_pcd.points = o3d.utility.Vector3dVector(vis_pair_pos[:, :3])
                        vis_pair_pos_pcd.colors = o3d.utility.Vector3dVector(vis_pair_pos[:, 3:])
                        vis_pair_pos_pcd = keypoints_to_spheres(vis_pair_pos_pcd, radius=0.1)

                        if vis_query_pos is not None:
                            vis_pos = np.concatenate([vis_query_pos, idx_color], axis=-1)  # (N_query, 6)
                            vis_pos_pcd = o3d.geometry.PointCloud()
                            vis_pos_pcd.points = o3d.utility.Vector3dVector(vis_pos[:, :3])
                            vis_pos_pcd.colors = o3d.utility.Vector3dVector(vis_pos[:, 3:])
                            vis_pos_pcd = keypoints_to_spheres(vis_pos_pcd, radius=0.1)

                            vis_tgt_scene = dense_scenes['pos'] + [vis_pos_pcd]
                            vis_ref_scene = dense_scenes['pair_pos'] + [vis_pair_pos_pcd]
                        else:
                            vis_tgt_scene = dense_scenes['pos']
                            vis_ref_scene = dense_scenes['pair_pos'] + [vis_pair_pos_pcd]
                    else:
                        raise NotImplementedError("Other input visualization modes not supported")

                    # Compute scale amounts for decent visualization
                    fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                    fp_ref_bounds = o3d_geometry_list_aabb(dense_scenes['pair_pos'])

                    fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                    fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                    resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
                    resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
                    vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                    vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                    # Compute shift amounts from bounding box
                    fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                    fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])

                    vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                    vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                    vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
                    vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

                    vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                    vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                    # Fix both scenes' ground level
                    tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])[1, 1]
                    ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])[1, 1]

                    vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
                    vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
                    geometry_list = vis_tgt_scene + vis_ref_scene
                    self.visualize_geometry(geometry_list, vis_sample_idx, scene_idx, pcd_resize_rate=(resize_ref_rate + resize_tgt_rate) / 2., save_prefix=f"{self.cfg.scene_pair_type}_{self.cfg.vis_input_mode}")

            # Optionally visualize global matches
            if args.vis_global_match_mode is not None:
                num_batches = self.scene_buffer_size // self.cfg.batch_size + 1 if self.scene_buffer_size % self.cfg.batch_size != 0 else self.scene_buffer_size // self.cfg.batch_size
                pair_pos_query_bounds = pair_pos_query.get_bounding_boxes().cpu().numpy()
                for scene_idx in range(num_batch_scenes):
                    vis_sample_idx = num_batches * num_batch_scenes * epoch + batch_idx * num_batch_scenes + scene_idx

                    if args.vis_global_match_mode == 'bbox_warp':
                        corner_idxs = np.array([
                            [0, 0, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, -1],
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [-1, -1, 0],
                            [-1, -1, -1]
                        ]).T  # Indices used for obtaining corners from query point boundaries
                        vis_pair_pos_corners = np.take_along_axis(pair_pos_query_bounds[scene_idx], corner_idxs, axis=-1).T  # (8, 3)

                        corner_lines = [[0, 1], [0, 2], [0, 4], [1, 3],
                                [1, 5], [2, 3], [2, 6], [3, 7],
                                [4, 5], [4, 6], [5, 7], [6, 7]]

                        # Use the same color for all lines
                        pair_pos_colors = [[1, 0, 0] for _ in range(len(corner_lines))]

                        pair_pos_line_set = o3d.geometry.LineSet()
                        pair_pos_line_set.points = o3d.utility.Vector3dVector(vis_pair_pos_corners)
                        pair_pos_line_set.lines = o3d.utility.Vector2iVector(corner_lines)
                        pair_pos_line_set.colors = o3d.utility.Vector3dVector(pair_pos_colors)
                        pair_pos_line_list = [pair_pos_line_set]

                        transform_line_list = []
                        for transform in global_transform_list[scene_idx]:
                            A_np, T_np = transform['A'].cpu().numpy(), transform['T'].cpu().numpy()
                            vis_warp_corners = vis_pair_pos_corners @ A_np.T + T_np
                            rand_color = [random.random(), random.random(), random.random()]
                            warp_colors = [rand_color for _ in range(len(corner_lines))]

                            warp_line_set = o3d.geometry.LineSet()
                            warp_line_set.points = o3d.utility.Vector3dVector(vis_warp_corners)
                            warp_line_set.lines = o3d.utility.Vector2iVector(corner_lines)
                            warp_line_set.colors = o3d.utility.Vector3dVector(warp_colors)
                            transform_line_list.append(warp_line_set)

                        dense_scenes = {}
                        dense_scenes['pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pos'][scene_idx], 'pos', remove_obj_mesh=remove_obj_mesh_from_full_scene)
                        dense_scenes['pair_pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pair_pos'][scene_idx], vis_pair_scene_name, remove_obj_mesh=remove_obj_mesh_from_full_scene)
                        dense_scenes['neg'] = build_dense_3d_scene(scene_pcd['scene_path']['neg'][scene_idx], 'neg', remove_obj_mesh=remove_obj_mesh_from_full_scene)

                        vis_tgt_scene = dense_scenes['pos'] + transform_line_list
                        vis_ref_scene = dense_scenes['pair_pos'] + pair_pos_line_list

                        # Compute scale amounts for decent visualization
                        fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                        fp_ref_bounds = o3d_geometry_list_aabb(dense_scenes['pair_pos'])

                        fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                        fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                        resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
                        resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]

                        vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                        vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                        # Compute shift amounts from bounding box
                        fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                        fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])

                        vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                        vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                        vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
                        vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

                        vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                        vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                        # Fix both scenes' ground level
                        tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:-1])[1, 1]
                        ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:-1])[1, 1]

                        vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
                        vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
                        geometry_list = vis_tgt_scene + vis_ref_scene
                        self.visualize_geometry(geometry_list, vis_sample_idx, scene_idx, pcd_resize_rate=(resize_ref_rate + resize_tgt_rate) / 2., save_prefix=f"{self.cfg.scene_pair_type}_{self.cfg.vis_global_match_mode}")

            # Optionally visualize local matches
            if args.vis_local_match_mode is not None:
                query_texture = [feat[..., 1:] for feat in query_pcd['pair_pos'].features_list()]
                query_inst_labels = [feat[..., 0].long() for feat in query_pcd['pair_pos'].features_list()]
                max_num_transform = self.cfg.local_topk  # Maximum number of transforms set manually

                for scene_idx in range(num_batch_scenes):
                    # Features are offloaded to CPU to save GPU memory
                    if self.cfg.vis_num_local_query != self.cfg.num_query:  # Re-generate local query points for visualization
                        inst_list = query_inst_labels[scene_idx].flatten().unique().tolist()
                        dense_query_pair_pos = extract_query_points_from_inst(scene_pcd['scene_path']['pair_pos'][scene_idx], inst_list, 'pair_pos', 
                            self.cfg.vis_num_local_query, load_color=(self.cfg.vis_local_match_mode == "texture_transfer"), point_sample_root=self.cfg.point_sample_root)
                        vis_query_pair_pos = dense_query_pair_pos.points_list()[0].cpu()
                        query_texture[scene_idx] = dense_query_pair_pos.features_list()[0][..., 1:]
                        query_inst_labels[scene_idx] = dense_query_pair_pos.features_list()[0][..., 0].long()
                    else:
                        vis_query_pair_pos = query_pcd['pair_pos'].points_list()[scene_idx].cpu()
                    vis_texture = query_texture[scene_idx].cpu()

                    num_query = vis_query_pair_pos.shape[0]

                    # Make color maps similar to dense semantic flow methods (https://github.com/kampta/asic)
                    color_wheel = get_color_wheel().numpy()
                    idx_color = map_coordinates_to_color(np.array(vis_query_pair_pos), color_wheel)

                    dense_scenes = {}
                    dense_scenes['pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pos'][scene_idx], 'pos', remove_obj_mesh=remove_obj_mesh_from_full_scene)
                    dense_scenes['pair_pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pair_pos'][scene_idx], vis_pair_scene_name, remove_obj_mesh=remove_obj_mesh_from_full_scene)
                    dense_scenes['neg'] = build_dense_3d_scene(scene_pcd['scene_path']['neg'][scene_idx], 'neg', remove_obj_mesh=remove_obj_mesh_from_full_scene)

                    # Visualize initial target points if there are no local transforms available
                    if len(local_transform_list[scene_idx]) == 0:
                        num_batches = self.scene_buffer_size // self.cfg.batch_size + 1 if self.scene_buffer_size % self.cfg.batch_size != 0 else self.scene_buffer_size // self.cfg.batch_size
                        vis_sample_idx = num_batches * max_num_transform * num_batch_scenes * epoch + batch_idx * max_num_transform * num_batch_scenes + \
                            scene_idx * max_num_transform

                        if args.vis_local_match_mode in ['intra_match', 'intra_sparse_match']:
                            vis_pair_pos = np.concatenate([vis_query_pair_pos, idx_color], axis=-1)  # (N_query, 6)
                        else:
                            vis_pair_pos = np.concatenate([vis_query_pair_pos, vis_texture], axis=-1)  # (N_query, 6)

                        vis_pair_pos_pcd = o3d.geometry.PointCloud()
                        vis_pair_pos_pcd.points = o3d.utility.Vector3dVector(vis_pair_pos[:, :3])
                        vis_pair_pos_pcd.colors = o3d.utility.Vector3dVector(vis_pair_pos[:, 3:])

                        vis_pair_pos_pcd = keypoints_to_spheres(vis_pair_pos_pcd, radius=0.1)

                        vis_tgt_scene = dense_scenes['pos']
                        vis_ref_scene = dense_scenes['pair_pos'] + [vis_pair_pos_pcd]

                        # Compute scale amounts for decent visualization
                        fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                        fp_ref_bounds = o3d_geometry_list_aabb(dense_scenes['pair_pos'])

                        fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                        fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                        resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
                        resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
                        vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                        vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                        # Compute shift amounts from bounding box
                        fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                        fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])

                        vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                        vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                        vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
                        vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

                        vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                        vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                        # Fix both scenes' ground level
                        tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])[1, 1]
                        ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])[1, 1]

                        vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
                        vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
                        geometry_list = vis_tgt_scene + vis_ref_scene
                        self.visualize_geometry(geometry_list, vis_sample_idx, scene_idx, pcd_resize_rate=(resize_ref_rate + resize_tgt_rate) / 2., save_prefix=f"{self.cfg.scene_pair_type}_{self.cfg.vis_local_match_mode}")

                    for t_idx, transform in enumerate(local_transform_list[scene_idx]):
                        vis_deform_query_pair_pos = transform(vis_query_pair_pos)
                        num_query = vis_deform_query_pair_pos.shape[0]

                        num_batches = self.scene_buffer_size // self.cfg.batch_size + 1 if self.scene_buffer_size % self.cfg.batch_size != 0 else self.scene_buffer_size // self.cfg.batch_size
                        vis_sample_idx = num_batches * max_num_transform * num_batch_scenes * epoch + batch_idx * max_num_transform * num_batch_scenes + \
                            scene_idx * max_num_transform + t_idx

                        # Build original dense 3D scene for visualization
                        use_pbr = args.vis_local_match_mode == 'region_2d_match_overlay'

                        if args.vis_local_match_mode in ['intra_match', 'texture_transfer']:
                            if args.vis_local_match_mode == 'intra_match':
                                vis_pair_pos = np.concatenate([vis_query_pair_pos, idx_color], axis=-1)  # (N_query, 6)
                                vis_deform_pair_pos = np.concatenate([vis_deform_query_pair_pos, idx_color], axis=-1)  # (N_query, 6)
                            else:
                                vis_pair_pos = np.concatenate([vis_query_pair_pos, vis_texture], axis=-1)  # (N_query, 6)
                                vis_deform_pair_pos = np.concatenate([vis_deform_query_pair_pos, vis_texture], axis=-1)  # (N_query, 6)

                            vis_pair_pos_pcd_o3d = o3d.geometry.PointCloud()
                            vis_pair_pos_pcd_o3d.points = o3d.utility.Vector3dVector(vis_pair_pos[:, :3])
                            vis_pair_pos_pcd_o3d.colors = o3d.utility.Vector3dVector(vis_pair_pos[:, 3:])
                            vis_deform_pair_pos_pcd_o3d = o3d.geometry.PointCloud()
                            vis_deform_pair_pos_pcd_o3d.points = o3d.utility.Vector3dVector(vis_deform_pair_pos[:, :3])
                            vis_deform_pair_pos_pcd_o3d.colors = o3d.utility.Vector3dVector(vis_deform_pair_pos[:, 3:])

                            vis_pair_pos_pcd = keypoints_to_spheres(vis_pair_pos_pcd_o3d, radius=0.1)
                            vis_deform_pair_pos_pcd = keypoints_to_spheres(vis_deform_pair_pos_pcd_o3d, radius=0.1)

                            vis_tgt_scene = dense_scenes['pos'] + [vis_deform_pair_pos_pcd]
                            vis_ref_scene = dense_scenes['pair_pos'] + [vis_pair_pos_pcd]

                            # Compute scale amounts for decent visualization
                            fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                            fp_ref_bounds = o3d_geometry_list_aabb(dense_scenes['pair_pos'])

                            fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                            fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                            resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
                            resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
                            vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                            vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                            # Compute shift amounts from bounding box
                            fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                            fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])

                            vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                            vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                            vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
                            vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

                            vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                            vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                            # Fix both scenes' ground level
                            tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])[1, 1]
                            ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])[1, 1]

                            vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
                            vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
                            geometry_list = vis_tgt_scene + vis_ref_scene
                            self.visualize_geometry(geometry_list, vis_sample_idx, scene_idx, pcd_resize_rate=(resize_ref_rate + resize_tgt_rate) / 2., save_prefix=f"{self.cfg.scene_pair_type}_{self.cfg.vis_local_match_mode}")

                            # Optionally save transfer results
                            if args.save_transfer:
                                for scene_name in ['pos', 'pair_pos']:
                                    transfer_save_root = os.path.join(self.mesh_save_dir, f"scene_{epoch}_{batch_idx}_{scene_idx}_{scene_name}")

                                    if not os.path.exists(transfer_save_root):
                                        os.makedirs(transfer_save_root, exist_ok=True)

                                    # Save query points
                                    if scene_name == 'pos':
                                        query_points_path = os.path.join(transfer_save_root, "query_points.ply")
                                        o3d.io.write_point_cloud(query_points_path, vis_deform_pair_pos_pcd_o3d)
                                    else:
                                        query_points_path = os.path.join(transfer_save_root, "query_points.ply")
                                        o3d.io.write_point_cloud(query_points_path, vis_pair_pos_pcd_o3d)

                                    # Save scene
                                    obj_count = 0
                                    for scene_entity in tqdm(dense_scenes[scene_name], desc=f"Saving {scene_name} scene {scene_idx}"):
                                        # add scene entities and floorplan linesets
                                        if isinstance(scene_entity, o3d.geometry.TriangleMesh):
                                            if len(scene_entity.textures) > 0:  # mesh
                                                entity_path = os.path.join(transfer_save_root, f"obj_{obj_count}.obj")
                                                o3d.io.write_triangle_mesh(entity_path, scene_entity)
                                            else:  # pointcloud
                                                entity_path = os.path.join(transfer_save_root, f"obj_{obj_count}.ply")
                                                o3d.io.write_triangle_mesh(entity_path, scene_entity)
                                            obj_count += 1
                                        elif isinstance(scene_entity, o3d.geometry.LineSet):
                                            fp_path = os.path.join(transfer_save_root, "fp.obj")  # .obj used to save color and texture
                                            lines = np.array(scene_entity.lines)
                                            vertices = np.array(scene_entity.points)

                                            triangulated = tr.triangulate({'vertices': vertices[:, [0, 2]], 'segments': lines}, 'p')
                                            triangles = triangulated['triangles']
                                            vertices = np.copy(triangulated['vertices'])  # new vertices are added if given vertices are not enough
                                            vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=1)[:, [0, 2, 1]]

                                            fp_model = o3d.geometry.TriangleMesh()
                                            fp_model.vertices = o3d.utility.Vector3dVector(vertices)
                                            fp_model.triangles = o3d.utility.Vector3iVector(triangles)
                                            fp_model.paint_uniform_color([0.8, 0.8, 0.8])

                                            uv = np.copy(vertices[:, [0, 2]])
                                            uv -= uv.min(axis=0)
                                            uv /= 1.0  # repeat every 1m
                                            texture_path = self.fp_texture_path

                                            tr_floor = trimesh.Trimesh(
                                                np.copy(vertices), np.copy(triangles), process=False
                                            )
                                            tr_floor.visual = trimesh.visual.TextureVisuals(
                                                uv=np.copy(uv),
                                                material=trimesh.visual.material.SimpleMaterial(
                                                    image=Image.open(texture_path)
                                                )
                                            )
                                            uvs = tr_floor.visual.uv
                                            triangles_uvs = []
                                            for i in range(3):
                                                triangles_uvs.append(uvs[tr_floor.faces[:, i]].reshape(-1, 1, 2))
                                            triangles_uvs = np.concatenate(triangles_uvs, axis=1).reshape(-1, 2)

                                            fp_model.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
                                            fp_model.textures = [o3d.geometry.Image(np.asarray(tr_floor.visual.material.image))]
                                            fp_model.triangle_material_ids = o3d.utility.IntVector([0] * len(tr_floor.faces))

                                            o3d.io.write_triangle_mesh(fp_path, fp_model)

                        elif args.vis_local_match_mode in ['intra_sparse_match']:
                            num_matches = vis_query_pair_pos.shape[0]
                            num_sample_matches = self.cfg.vis_match_sample_num
                            vis_match_idx = choice_without_replacement(np.arange(num_matches, dtype=int), n=num_sample_matches)
                            vis_match_idx = np.sort(vis_match_idx)
                            match_colors = np.linspace(0., 1., num_sample_matches)  # (N_query, )
                            match_colors = colormaps['jet'](match_colors, alpha=False, bytes=False)[:, :3]

                            vis_pair_pos = np.concatenate([vis_query_pair_pos[vis_match_idx], match_colors], axis=-1)  # (N_query, 6)
                            vis_deform_pair_pos = np.concatenate([vis_deform_query_pair_pos[vis_match_idx], match_colors], axis=-1)  # (N_query, 6)

                            vis_pair_pos_pcd = o3d.geometry.PointCloud()
                            vis_pair_pos_pcd.points = o3d.utility.Vector3dVector(vis_pair_pos[:, :3])
                            vis_pair_pos_pcd.colors = o3d.utility.Vector3dVector(vis_pair_pos[:, 3:])
                            vis_deform_pair_pos_pcd = o3d.geometry.PointCloud()
                            vis_deform_pair_pos_pcd.points = o3d.utility.Vector3dVector(vis_deform_pair_pos[:, :3])
                            vis_deform_pair_pos_pcd.colors = o3d.utility.Vector3dVector(vis_deform_pair_pos[:, 3:])

                            vis_pair_pos_centroid = o3d.geometry.PointCloud(vis_pair_pos_pcd)
                            vis_deform_pair_pos_centroid = o3d.geometry.PointCloud(vis_deform_pair_pos_pcd)
                            vis_pair_pos_pcd = keypoints_to_spheres(vis_pair_pos_pcd, radius=0.15)
                            vis_deform_pair_pos_pcd = keypoints_to_spheres(vis_deform_pair_pos_pcd, radius=0.15)

                            vis_tgt_scene = dense_scenes['pos'] + [vis_deform_pair_pos_pcd, vis_deform_pair_pos_centroid]
                            vis_ref_scene = dense_scenes['pair_pos'] + [vis_pair_pos_pcd, vis_pair_pos_centroid]

                            # Compute scale amounts for decent visualization
                            fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                            fp_ref_bounds = o3d_geometry_list_aabb(dense_scenes['pair_pos'])

                            fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                            fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                            resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
                            resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
                            vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                            vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                            # Compute shift amounts from bounding box
                            fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                            fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])

                            vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                            vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                            vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
                            vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

                            vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                            vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                            # Fix both scenes' ground level
                            tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])[1, 1]
                            ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])[1, 1]

                            vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
                            vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])

                            # Add lines for sparse matches
                            match_line_set = o3d.geometry.LineSet()
                            match_line_set.points = o3d.utility.Vector3dVector(np.concatenate([
                                np.asarray(vis_tgt_scene[-1].points), np.asarray(vis_ref_scene[-1].points)
                            ], axis=0))
                            match_line_set.lines = o3d.utility.Vector2iVector(np.stack([np.arange(num_sample_matches), np.arange(num_sample_matches) + num_sample_matches], axis=-1))
                            match_line_set.colors = o3d.utility.Vector3dVector(match_colors)

                            geometry_list = vis_tgt_scene + vis_ref_scene + [match_line_set]
                            self.visualize_geometry(geometry_list, vis_sample_idx, scene_idx, pcd_resize_rate=(resize_ref_rate + resize_tgt_rate) / 2., save_prefix=f"{self.cfg.scene_pair_type}_{self.cfg.vis_local_match_mode}")

                        elif args.vis_local_match_mode in ['region_2d_match', 'region_2d_match_overlay']:
                            if t_idx == 0:  # Set pair_pos_region_2d fixed during visualization
                                pair_pos_region_2d = generate_random_region_2d(query_pcd['pair_pos'][scene_idx], num_grid_points=self.cfg.vis_region_2d_num_grid, len_range=self.cfg.vis_region_2d_len_range)
                                vis_region_2d_pair_pos = pair_pos_region_2d.points_padded()[0].cpu()
                            vis_deform_region_2d_pair_pos = transform(vis_region_2d_pair_pos)

                            num_region_2d = vis_region_2d_pair_pos.shape[0]
                            region_2d_idx_color = np.linspace(0., 1., num_region_2d)  # (N_query, )
                            region_2d_idx_color = colormaps['jet'](region_2d_idx_color, alpha=False, bytes=False)[:, :3]

                            vis_pair_pos = np.concatenate([vis_region_2d_pair_pos.cpu(), region_2d_idx_color], axis=-1)  # (N_query, 6)
                            vis_deform_pair_pos = np.concatenate([vis_deform_region_2d_pair_pos.cpu(), region_2d_idx_color], axis=-1)  # (N_query, 6)

                            vis_pair_pos_pcd = o3d.geometry.PointCloud()
                            vis_pair_pos_pcd.points = o3d.utility.Vector3dVector(vis_pair_pos[:, :3])
                            vis_pair_pos_pcd.colors = o3d.utility.Vector3dVector(vis_pair_pos[:, 3:])
                            vis_deform_pair_pos_pcd = o3d.geometry.PointCloud()
                            vis_deform_pair_pos_pcd.points = o3d.utility.Vector3dVector(vis_deform_pair_pos[:, :3])
                            vis_deform_pair_pos_pcd.colors = o3d.utility.Vector3dVector(vis_deform_pair_pos[:, 3:])

                            vis_pair_pos_pcd = keypoints_to_spheres(vis_pair_pos_pcd, radius=0.1)
                            vis_deform_pair_pos_pcd = keypoints_to_spheres(vis_deform_pair_pos_pcd, radius=0.1)

                            vis_tgt_scene = dense_scenes['pos'] + [vis_deform_pair_pos_pcd]
                            vis_ref_scene = dense_scenes['pair_pos'] + [vis_pair_pos_pcd]

                            # Compute scale amounts for decent visualization
                            fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                            fp_ref_bounds = o3d_geometry_list_aabb(dense_scenes['pair_pos'])

                            fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                            fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                            resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
                            resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
                            vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                            vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                            # Compute shift amounts from bounding box
                            fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
                            fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])

                            vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                            vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                            vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
                            vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

                            vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                            vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                            # Fix both scenes' ground level
                            tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len(dense_scenes['pos'])])[1, 1]
                            ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len(dense_scenes['pair_pos'])])[1, 1]

                            vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
                            vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
                            geometry_list = vis_tgt_scene + vis_ref_scene

                            if args.vis_local_match_mode == 'region_2d_match_overlay':
                                pbr_geometry_list = []

                                for geo_idx, geom in enumerate(geometry_list):
                                    if geo_idx in [len(vis_tgt_scene) - 1, len(vis_tgt_scene) + len(vis_ref_scene) - 1]:  # Make original query points translucent
                                        material = o3d.visualization.rendering.MaterialRecord()
                                        material.shader = 'defaultLitTransparency'
                                        material.base_color = [0.4, 0.4, 0.4, 0.7]  # RGBA, adjust A for transparency
                                    else:
                                        if getattr(geom, 'textures', None) is not None:
                                            material = o3d.visualization.rendering.MaterialRecord()
                                            material.shader = 'defaultUnlit'
                                            material.albedo_img = o3d.geometry.Image(geom.textures[0])
                                        else:
                                            material = None

                                    pbr_geometry_list.append({'name': f'geom_{geo_idx}', 'geometry': geom, 'material': material})

                                # NOTE: Currently this only supports screen display (TODO: Have this to support image saving)
                                assert self.cfg.vis_method == 'screen'

                                self.visualize_geometry(pbr_geometry_list, vis_sample_idx, scene_idx, use_pbr=use_pbr, save_prefix=f"{self.cfg.scene_pair_type}_{self.cfg.vis_local_match_mode}")
                            else:
                                self.visualize_geometry(geometry_list, vis_sample_idx, scene_idx, use_pbr=use_pbr, save_prefix=f"{self.cfg.scene_pair_type}_{self.cfg.vis_local_match_mode}")

    def visualize_geometry(self, geometry_list, vis_sample_idx=0, scene_idx=0, repeat_idx=0, num_repeats=1, close_window_at_end=False, pcd_resize_rate=1., use_pbr=False, save_prefix="default"):  # Helper function for visualizing each scene in batch
        # vis_sample_idx is the visualization index to use for saving rendered image / video
        # scene_idx is index of scene within batch, repeat_idx is the index of the repeat in repeated visualization (e.g., feat_dist visualization)
        # NOTE: Indices only matter for render_img and render_video modes

        if self.cfg.vis_method == "screen":
            if use_pbr:
                o3d.visualization.draw(geometry_list, show_skybox=False)
            else:
                o3d.visualization.draw_geometries(geometry_list)
        elif self.cfg.vis_method in ["render_img", "render_video", "render_img_top_down"]:
            if not os.path.exists(os.path.join(self.cfg.log_dir, f'{save_prefix}_rendering')):
                os.makedirs(os.path.join(self.cfg.log_dir, f'{save_prefix}_rendering'), exist_ok=True)

            # Initialize open3d visualizer for new scene
            if repeat_idx == 0:  # Generate window only for first frame
                self.visualizer = o3d.visualization.Visualizer()
                self.visualizer.create_window()
                self.visualizer.get_render_option().point_size = self.cfg.vis_point_size * pcd_resize_rate

            for geometry in geometry_list:
                self.visualizer.add_geometry(geometry)
                self.visualizer.update_geometry(geometry)

            # Change to top-down view
            ctr = self.visualizer.get_view_control()
            rot = np.eye(4)
            rot[:3, :3] = R.from_euler('x', 80, degrees=True).as_matrix()
            cam = ctr.convert_to_pinhole_camera_parameters()
            cam.extrinsic = cam.extrinsic @ rot

            # Fix camera parameters to near-orthographic
            new_cam_extrinsic = np.copy(cam.extrinsic)
            new_cam_extrinsic[0, -1] = 0.  # Make camera centered around scene
            new_cam_extrinsic[1, -1] = 0.  # Make camera centered around scene
            new_cam_extrinsic[2, -1] += 300.
            cam.extrinsic = new_cam_extrinsic
            new_cam_intrinsic = np.copy(cam.intrinsic.intrinsic_matrix)
            new_cam_intrinsic[0, 0] *= 20.
            new_cam_intrinsic[1, 1] *= 20.
            cam.intrinsic.intrinsic_matrix = new_cam_intrinsic
            ctr.convert_from_pinhole_camera_parameters(cam, True)

            self.visualizer.poll_events()
            self.visualizer.update_renderer()

            if self.cfg.vis_method == "render_img":
                self.visualizer.capture_screen_image(os.path.join(self.cfg.log_dir, f'{save_prefix}_rendering', f'render_{vis_sample_idx}_repeat_{repeat_idx}.png'))
            elif self.cfg.vis_method == "render_img_top_down":
                self.visualizer.capture_screen_image(os.path.join(self.cfg.log_dir, f'{save_prefix}_rendering', f'render_top_{vis_sample_idx}_repeat_{repeat_idx}.png'))
                ctr = self.visualizer.get_view_control()
                rot = np.eye(4)
                rot[:3, :3] = R.from_euler('x', -180, degrees=True).as_matrix()
                cam = ctr.convert_to_pinhole_camera_parameters()
                cam.extrinsic = cam.extrinsic @ rot
                ctr.convert_from_pinhole_camera_parameters(cam, True)
                ctr.set_zoom(0.4)

                self.visualizer.poll_events()
                self.visualizer.update_renderer()
                self.visualizer.capture_screen_image(os.path.join(self.cfg.log_dir, f'{save_prefix}_rendering', f'render_down_{vis_sample_idx}_repeat_{repeat_idx}.png'))
            else:  # Render to video
                frame = self.visualizer.capture_screen_float_buffer()
                frame = (255 * np.asarray(frame)).astype(np.uint8)

                if repeat_idx == 0:
                    self.video = cv2.VideoWriter(
                        os.path.join(self.cfg.log_dir, f'{save_prefix}_rendering', f'render_{vis_sample_idx}' + '.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        5.,
                        (frame.shape[1], frame.shape[0]))

                self.video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if repeat_idx == num_repeats - 1:
                    self.video.release()

            self.visualizer.clear_geometries()

            repeat_idx += 1
            if close_window_at_end:
                self.visualizer.destroy_window()
        else:
            raise NotImplementedError("Other visualization methods not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General configs
    parser.add_argument("--log_dir", help="Log directory for saving experiment results", default="./log/")
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)
    parser.add_argument("--wandb", help="Optionally log metrics to wandb", action='store_true')
    parser.add_argument("--run_name", help="Optional nametag for run logging in wandb", default=None, type=str)

    # Dataset configs
    parser.add_argument("--scene_sample_points", type=int, help="Number of points to sample per object model for scene generation", default=50)
    parser.add_argument("--point_sample_root", help="Root folder containing point samples from object mesh files (currently used for scene generation without feature extraction)", default="./data/arkit_scenes_point_samples/")
    parser.add_argument("--random_group_topk_range", type=int, help="Minimum and maximum value for top-K nearest object sampling during random group query sampling", default=[1, 2], nargs=2)
    parser.add_argument("--scene_pair_type", type=str, help="Type of scene pairs to load for matching", default="identical")
    parser.add_argument("--obj_list_file", default="./data/arkit_scenes_gen_assets/obj_dirs.txt", type=str, help=".txt file containing paths to object meshes")
    parser.add_argument("--obj_root", default="./data/arkit_scenes_processed/objects/", type=str)
    parser.add_argument("--scene_pair_file", type=str, help="Path to .tsv file containing list of scenes with ground-truth matching objects", default=None)

    # Training configs
    parser.add_argument("--eval_reps", help="Number of evaluation repititions to make", default=1, type=int)
    parser.add_argument("--mode", help="Mode to use for evaluation", default="regular", type=str)
    parser.add_argument("--scene_root", help="Root directory containing scene data", required=True, type=str)
    parser.add_argument("--scene_augment_root", help="Root directory containing augmented scene data used when scene_pair_type is 'augment'", default=None, type=str)
    parser.add_argument("--batch_size", help="Batch size to use for training", default=4, type=int)
    parser.add_argument("--scene_buffer_size", help="Size of buffer to use for loading scenes each epoch", default=32, type=int)
    parser.add_argument("--query_point_type", help="Type of query point sampling to use for generating trianing samples", default="obj_surface_points", type=str)
    parser.add_argument("--num_vert_split", help="Number of vertical splits to make for floorplan queries", default=1, type=int)
    parser.add_argument("--force_up_margin", help="Forced upper margin for floorplan queries (defaults to using maximum height of walls)", default=0.5, type=float)
    parser.add_argument("--force_low_margin", help="Forced lower margin for floorplan queries (defaults to using minimum height of walls)", default=0.5, type=float)
    parser.add_argument("--num_query", help="Number of query point locations to exploit per scene for local feature extraction", default=50, type=int)
    parser.add_argument("--vis_num_local_query", help="Number of query point locations to exploit per scene for visualizing local matching", default=50, type=int)
    parser.add_argument("--update_buffer_every", help="Number of epochs before updating buffer", default=1, type=int)
    parser.add_argument("--num_classes", help="Number of semantics classes in objects", default=17, type=int)
    parser.add_argument("--fp_feat_type", help="Type of floorplan features to use", default="learned", type=str)
    parser.add_argument("--obj_point_query_scale_factor", help="Scale factor to use for object point query sampling", default=None, type=float, nargs="+")
    parser.add_argument("--query_sampling_method", help="Type of query sampling to use for training deformation field", default="region", type=str)
    parser.add_argument("--region_scale_range", help="Minimum / maximum range values for generating sampling regions", default=[0.5, 2.5], type=float, nargs="+")
    parser.add_argument("--pair_pos_bbox_num_grid_points", help="Number of grid points to sample per axis for positive bbox to be matched", type=int, default=5)
    parser.add_argument("--align_height", help="Optionally align height values of initial search query points", action="store_true")
    parser.add_argument("--global_match_valid_thres", help="Valid threshold for keeping instance match based on an initial global affine transform", type=float, default=2.)
    parser.add_argument("--global_dist_valid_thres", help="Valid threshold for keeping a global feature distance-based transform", type=float, default=1.5)
    parser.add_argument("--global_num_rot", help="Number of rotation splits (along yaw) for initialization", type=int, default=16)
    parser.add_argument("--global_num_iter", help="Number of iterations for global map estimation", type=int, default=100)
    parser.add_argument("--global_topk", help="Top-k number of locations to consider for global map estimation", type=int, default=30)
    parser.add_argument("--global_lr", help="Optimization learning rate (step size) for global map estimation", type=float, default=0.001)
    parser.add_argument("--global_patience", help="Patience value for learning rate scheduling in global map estimation", type=int, default=5)
    parser.add_argument("--global_factor", help="Learning rate decay factor for global map estimation", type=float, default=0.9)
    parser.add_argument("--global_cost_type", help="Type of cost function to use for global map estimation", type=str, default="l2")
    parser.add_argument("--global_mapping_type", help="Type of mapping to use for global map estimation", type=str, default="scale_rot")
    parser.add_argument("--global_nms_type", help="Type of non-maximum suppression to apply during global matching", type=str, default="bbox")
    parser.add_argument("--global_nms_thres", help="Threshold value for non-maximum suppression in global matching", type=float, default=0.5)
    parser.add_argument("--local_num_iter", help="Number of iterations for local map estimation", type=int, default=100)
    parser.add_argument("--local_lr", help="Optimization learning rate (step size) for local map estimation", type=float, default=0.001)
    parser.add_argument("--local_skip_box_align", help="Optionally skip bounding box alignment prior to local displacement mapping", action="store_true")
    parser.add_argument("--local_patience", help="Patience value for learning rate scheduling in local map estimation", type=int, default=5)
    parser.add_argument("--local_factor", help="Learning rate decay factor for local map estimation", type=float, default=0.9)
    parser.add_argument("--local_cost_type", help="Type of cost function to use for local map estimation", type=str, default="l2")
    parser.add_argument("--local_rbf_smoothing", help="RBF smoothing parameter for local map estimation", type=float, default=0.5)
    parser.add_argument("--local_dist_cost_weight", help="Weight value for distance cost during local map estimation", type=float, default=0.0)
    parser.add_argument("--local_feat_cost_weight", help="Weight value for feature cost during local map estimation", type=float, default=1.0)
    parser.add_argument("--local_topk", help="Top-k number of locations to consider for local map estimation", type=int, default=1)
    parser.add_argument("--local_valid_thres", help="Valid threshold for keeping a local transformation", type=float, default=2.0)
    parser.add_argument("--local_nms_thres", help="Threshold value for non-maximum suppression in local matching", type=float, default=0.5)
    parser.add_argument("--skip_local_matching", help="If set, skips local matching process", action="store_true", default=False)
    parser.add_argument("--local_matcher_type", help="Type of local matcher to use for establishing fine matches", type=str, default="point")
    parser.add_argument("--global_matcher_type", help="Type of global matcher to use for establishing coarse matches", type=str, default="affine")
    parser.add_argument("--fp_point_type", help="Type of floorplan points to use", default="wireframe", type=str)
    parser.add_argument("--fp_sample_step_size", help="Step size (contour, height) for generating floorplan sampled points", default=[0.3, 0.3], type=float, nargs=2)
    parser.add_argument("--fp_label_type", help="Type of floorplan labeling to use", default="single", type=str)
    parser.add_argument("--query_obj_match_mode", help="Type of solver to use for obtaining object match points during query sampling when query_point_type is obj_match_points", default="jv", type=str)
    parser.add_argument("--query_obj_match_add_bbox", help="Optionally add bounding box proximal points during query sampling when query_point_type is obj_match_points", action="store_true")
    parser.add_argument("--save_scene_mesh", help="Optionally save scene meshes in log directory", action="store_true")
    parser.add_argument("--save_transfer", help="Optionally save query point transfer in log directory", action="store_true")
    parser.add_argument("--map_acc_thres", help="List of threshold values to use when computing map accuracy", default=[0.25, 0.5, 0.75], type=float, nargs="+")
    parser.add_argument("--bijectivity_acc_thres", help="List of threshold values to use when computing bijectivity accuracy", default=[0.25, 0.5, 0.75], type=float, nargs="+")
    parser.add_argument("--chamfer_acc_thres", help="List of threshold values to use when Chamfer accuracy", default=[0.15, 0.2, 0.25], type=float, nargs="+")

    # Feature field configs
    parser.add_argument("--load_local_feature_field", help="Load pre-trained local feature field", default=None, type=str)
    parser.add_argument("--load_global_feature_field", help="Load pre-trained global feature field", default=None, type=str)
    parser.add_argument("--max_infer_point", help="Maximum number of query points to process per inference", default=1000, type=int)

    # Visualization configs
    parser.add_argument("--vis_input_mode", help="Mode for visualizing input scene pairs", default=None, type=str)
    parser.add_argument("--vis_global_match_mode", help="Mode for visualizing global matches during evaluation", default=None, type=str)
    parser.add_argument("--vis_local_match_mode", help="Mode for visualizing local matches during evaluation", default=None, type=str)
    parser.add_argument("--vis_margin", help="Amount of margins to apply for reference scene during visualization", default=5., type=float)
    parser.add_argument("--vis_method", help="Type of visualization to perform", default="screen", type=str)
    parser.add_argument("--vis_sort_query", help="Type of sorting to perform on query points", default='xyz', type=str)
    parser.add_argument("--vis_region_2d_num_grid", help="Number of grid points per axis when generating random 2D regions", default=50, type=int)
    parser.add_argument("--vis_region_2d_len_range", help="Range of random 2D region sizes to generate", default=[1., 2.], type=float, nargs="+")
    parser.add_argument("--vis_match_sample_num", help="Number of dense matches to visualize for local dense matching", default=5, type=int)
    parser.add_argument("--vis_point_size", help="Size of point for rendering views", default=2., type=float)
    parser.add_argument("--fp_texture_path", help="Path for floorplan texture", default="data/sample_textures/texture_uniform4.png", type=str)

    # Evaluation configs
    args = parser.parse_args()

    # Fix seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Optionally initialize wandb for logging
    if args.wandb:
        wandb.init(project="neu-rel-field", config=vars(args), name=args.run_name)

    evaluator = RelMatchEvaluator(cfg=args, log_dir=args.log_dir)
    evaluator.run()
