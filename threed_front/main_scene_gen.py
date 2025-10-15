import argparse
import torch
import numpy as np
import random
import os
from glob import glob
from tqdm import trange
import open3d as o3d
import json
from threed_front.atiss_utils.threed_front_dataset import ThreedFront
import trimesh
from PIL import Image
from threed_front.scene_gen_utils import floor_plan_from_scene
import copy
from threed_front.utils import (
    contour_from_floorplan_mesh,
    choice_without_replacement,
    RollingSampler,
    keypoints_to_spheres,
    str2cls_3dfuture
)
from tqdm import tqdm
import pickle
from threed_front.invalid_files import PROBLEMATIC_OBJ_ID, PROBLEMATIC_SCENE_ID
from threed_front.utils import trimesh_load_with_postprocess
import subprocess

SUBCLASS_MIN_COUNT = 5  # Minimum number of objects in subclasses to be considered usable


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_root", default="./data/3D-FUTURE-model/", type=str)
    parser.add_argument("--scene_root", help="Root path to .json files specifying scenes for inspection", default="./data/3D-FRONT/", type=str)
    parser.add_argument("--fp_texture_root", help="Root path to floorplan texture files", default="./data/3D-FRONT-texture/", type=str)
    parser.add_argument("--obj_model_info_path", help="Path to .json file specifying object for inspection", default="./data/3D-FUTURE-model/model_info.json", type=str)
    parser.add_argument("--num_scenes", help="Total number of scenes to generate", default=-1, type=int)
    parser.add_argument("--num_load_layout", help="Number of floor-scale scene paths to load prior to visualization", type=int, default=-1)
    parser.add_argument("--save_root", help="Directory for saving generated scnees", default="./data/3d_front_scenes/", type=str)
    parser.add_argument("--aspect_ratio_path", help="Path to saving object aspect ratios", default="./data/3d_front_gen_assets/obj_aspect_ratio.pkl", type=str)
    parser.add_argument("--bbox_corner_path", help="Path to saving bounding box corners", default="./data/3d_front_gen_assets/obj_bbox_corner.pkl", type=str)
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)
    parser.add_argument("--max_trans_noise", help="Maximum noise level for translation", default=0.1, type=float)
    parser.add_argument("--max_rot_noise", help="Maximum noise level for rotation in degrees", default=22.5, type=float)
    parser.add_argument("--max_fp_noise", help="Maximum noise level for floorplan perturbation", default=0.2, type=float)
    parser.add_argument("--floorplan_type", help="Type of floorplan to use for data saving", default="two_level_lines", type=str)
    parser.add_argument("--visualize_scene", help="Optionally visualize scene", action="store_true")
    parser.add_argument("--default_scene_height", type=float, help="Default scene height to use for scene generation", default=3.0)
    parser.add_argument("--rescaling_method", type=str, help="Type of re-scaling to perform on replaced objects", default="box_scale")
    parser.add_argument("--train_ratio", type=float, help="Ratio of total rooms to use for training", default=0.90)
    parser.add_argument("--skip_object_splitting", help="Optionally split object splitting for train / test", action="store_true")
    parser.add_argument("--topk_aspect_sampling", help="Top-k objects according to aspect ratio distance to sample from", type=int, default=100)
    parser.add_argument("--ignore_subclass", help="Optionally ignore subclass labels during object replacement", action="store_true")
    args = parser.parse_args()

    # Fix seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Parse repository of object shapes
    obj_dirs = [obj for obj in sorted(glob(os.path.join(args.obj_root, "*"))) if ('.json' not in obj) and ('.py' not in obj)]
    num_objs = len(obj_dirs)

    # Post-process 3D-FUTURE object directory
    with open(args.obj_model_info_path, 'r') as f:
        obj_list = json.load(f)

    # Remove invalid objects
    obj_list = [obj for obj in obj_list if obj['model_id'] not in PROBLEMATIC_OBJ_ID]

    obj_classes = []
    cls2obj_id = {}
    subcls2obj_id = {}
    obj_id2class = {}
    obj_id2subclass = {}

    # Load mapping from object ID to aspect ratio
    if os.path.exists(args.aspect_ratio_path):
        caching_aspect_ratio = False
        with open(args.aspect_ratio_path, 'rb') as f:
            obj_id2aspect_vec = pickle.load(f)
    else:
        aspect_ratio_dir = args.aspect_ratio_path.replace("obj_aspect_ratio.pkl", "")
        if not os.path.exists(aspect_ratio_dir):
            os.makedirs(aspect_ratio_dir, exist_ok=True)
        caching_aspect_ratio = True
        obj_id2aspect_vec = {}

    # Load mapping from object ID to bounding box corners
    if os.path.exists(args.bbox_corner_path):
        caching_bbox_corner = False
        with open(args.bbox_corner_path, 'rb') as f:
            obj_id2bbox_corner = pickle.load(f)
    else:
        bbox_corner_dir = args.bbox_corner_path.replace("obj_bbox_corner.pkl", "")
        if not os.path.exists(bbox_corner_dir):
            os.makedirs(bbox_corner_dir, exist_ok=True)
        caching_bbox_corner = True
        obj_id2bbox_corner = {}

    for obj_dict in tqdm(obj_list, desc="Parsing object info"):
        obj_class = obj_dict['super-category'].lower()
        obj_classes.append(obj_class)

        if obj_class not in cls2obj_id.keys():
            cls2obj_id[obj_class] = []
        cls2obj_id[obj_class].append(obj_dict['model_id'])
        obj_id2class[obj_dict['model_id']] = obj_class

        if obj_dict.get('category', None) is not None:
            obj_subclass = obj_dict['category'].lower()
            obj_id2subclass[obj_dict['model_id']] = obj_subclass
            if obj_subclass not in subcls2obj_id.keys():
                subcls2obj_id[obj_subclass] = []
            subcls2obj_id[obj_subclass].append(obj_dict['model_id'])

        if caching_aspect_ratio or caching_bbox_corner:
            # Compute aspect ratio
            obj_path = os.path.join(args.obj_root, obj_dict['model_id'], "raw_model.obj")
            tr_mesh = trimesh_load_with_postprocess(obj_path, postprocess_type='bottom_crop')
            bbox_size = np.asarray(tr_mesh.vertices).max(0) - np.asarray(tr_mesh.vertices).min(0)
            aspect_vec = bbox_size / bbox_size[0]
            bbox_corners = np.stack([
                np.asarray(tr_mesh.vertices).max(0),
                np.asarray(tr_mesh.vertices).min(0)
            ], axis=0)  # (2, 3)
            if caching_aspect_ratio:
                obj_id2aspect_vec[obj_dict['model_id']] = aspect_vec
            if caching_bbox_corner:
                obj_id2bbox_corner[obj_dict['model_id']] = bbox_corners

    # Compute statistics for subclasses
    subcls_count = {subcls: len(subcls2obj_id[subcls]) for subcls in subcls2obj_id.keys()}
    num_subcls_objs = sum([subcls_count[subcls] for subcls in subcls_count.keys()])

    # Cache obj_id2aspect_vec as this takes quite a while to compute
    if caching_aspect_ratio:
        with open(args.aspect_ratio_path, 'wb') as f:
            pickle.dump(obj_id2aspect_vec, f)

    # Cache obj_id2bbox_corner as this takes quite a while to compute
    if caching_bbox_corner:
        with open(args.bbox_corner_path, 'wb') as f:
            pickle.dump(obj_id2bbox_corner, f)

    obj_classes = list(set(obj_classes))

    # Parse rooms in 3D-FRONT
    threed_rooms = ThreedFront.from_dataset_directory(
        args.scene_root,
        args.obj_model_info_path,
        args.obj_root,
        num_load_layout=args.num_load_layout
    )
    valid_threed_rooms_scenes = [scene for scene in threed_rooms.scenes if scene.uid not in PROBLEMATIC_SCENE_ID]
    print(f"Number of rooms: {len(valid_threed_rooms_scenes)}")

    # NOTE: The number of scenes can be larger than the actual number of scenes in 3D-FRONT
    num_scenes = args.num_scenes if args.num_scenes != -1 else len(valid_threed_rooms_scenes)

    if not args.skip_object_splitting:
        num_train_scenes = int(num_scenes * args.train_ratio)
        num_test_scenes = num_scenes - num_train_scenes

        # Identify scenes and objects to use for training
        test_scenes, test_scenes_idx = choice_without_replacement(valid_threed_rooms_scenes, len(valid_threed_rooms_scenes) - int(len(valid_threed_rooms_scenes) * args.train_ratio), return_idx=True)
        train_scenes = [scene for s_idx, scene in enumerate(valid_threed_rooms_scenes) if s_idx not in test_scenes_idx]

        test_scene_sampler = RollingSampler(test_scenes)
        train_scene_sampler = RollingSampler(train_scenes)

        test_obj_id_list = []
        for scene in test_scenes:
            for obj in scene.bboxes:
                obj_id = obj.model_jid
                test_obj_id_list.append(obj_id)

        # Make separate object mappings for train and test
        train_cls2obj_id = {}
        test_cls2obj_id = {}
        train_obj_id_list = []
        for cls_name in cls2obj_id.keys():
            train_cls2obj_id[cls_name] = []
            test_cls2obj_id[cls_name] = []
            for obj_id in cls2obj_id[cls_name]:
                if obj_id not in PROBLEMATIC_OBJ_ID:  # Remove object ID that are problematic
                    if obj_id in test_obj_id_list:
                        test_cls2obj_id[cls_name].append(obj_id)
                    else:
                        train_cls2obj_id[cls_name].append(obj_id)
                        train_obj_id_list.append(obj_id)

        # Make additional mappings for subclasses
        train_subcls2obj_id = {}
        test_subcls2obj_id = {}
        for subcls_name in subcls2obj_id.keys():
            train_subcls2obj_id[subcls_name] = []
            test_subcls2obj_id[subcls_name] = []
            for obj_id in subcls2obj_id[subcls_name]:
                if obj_id not in PROBLEMATIC_OBJ_ID:  # Remove object ID that are problematic
                    if obj_id in test_obj_id_list:
                        test_subcls2obj_id[subcls_name].append(obj_id)
                    else:
                        train_subcls2obj_id[subcls_name].append(obj_id)

    # Note the scene scales follow the object point cloud scales
    for scene_idx in trange(num_scenes):
        if not args.skip_object_splitting:
            if scene_idx < num_train_scenes:
                scene_mode = 'train'
            else:
                scene_mode = 'test'

            # Scenes will be generated in a roullette-based fashion
            if scene_mode == 'train':
                scene = train_scene_sampler.sample(1)[0]
            else:
                scene = test_scene_sampler.sample(1)[0]
        else:
            scene_mode = None
            scene = valid_threed_rooms_scenes[scene_idx % len(valid_threed_rooms_scenes)]  # Scenes will be generated in a roullette-based fashion

        # Sanity-check objects in scene
        if any([obj.model_jid in PROBLEMATIC_OBJ_ID for obj in scene.bboxes]):
            print("Invalid object encountered: skipping!")
            continue

        # Load objects from original scene
        orig_scene_model = {
            'floorplan': None,
            'bboxes': [],  # Bboxes are scale transformed
            'trans': [],
            'rot': [],
            'bbox_centroids': [],  # Centroid coordinates of scale-transformed raw object model bounding boxes
            'obj_classes': [],
            'obj_classes_str': [],
            'obj_subclasses_str': [],
            'obj_path': [],
            'obj_id': [],
            'obj_scene_scales': []  # Scale values of objects within each scene (pre-multiplied to 'objects' point cloud)
        }

        orig_swap_dict = {}
        for obj in scene.bboxes:
            obj_id = obj.model_jid
            obj_model_path = os.path.join(args.obj_root, obj_id, "raw_model.obj")

            obj_bbox_corners = obj_id2bbox_corner[obj_id] * obj.scale
            obj_bbox_sizes = obj_bbox_corners[0] - obj_bbox_corners[1]
            obj_centroids = (obj_bbox_corners[0] + obj_bbox_corners[1]) / 2.
            theta = -obj.z_angle  # NOTE: The signs are inverted to use the X' = X @ R.T + T convention
            R = np.zeros((3, 3))
            R[0, 0] = np.cos(theta)
            R[0, 2] = -np.sin(theta)
            R[2, 0] = np.sin(theta)
            R[2, 2] = np.cos(theta)
            R[1, 1] = 1.

            orig_scene_model['bbox_centroids'].append(obj_centroids)
            orig_scene_model['bboxes'].append(obj_bbox_sizes)
            orig_scene_model['trans'].append(np.array(obj.position) - np.array(scene.centroid))
            orig_scene_model['rot'].append(R)

            # Load object classes
            cls_label = obj_id2class[obj_id]
            cls_label_id = str2cls_3dfuture(cls_label, return_type='int')
            if obj_id in obj_id2subclass.keys():
                subcls_label = obj_id2subclass[obj_id]
            else:  # Use class label to represent labels
                subcls_label = obj_id2class[obj_id]
            orig_scene_model['obj_classes'].append(cls_label_id)
            orig_scene_model['obj_classes_str'].append(cls_label)
            orig_scene_model['obj_subclasses_str'].append(subcls_label)
            orig_scene_model['obj_path'].append(obj_model_path)
            orig_scene_model['obj_id'].append(obj_id)
            orig_scene_model['obj_scene_scales'].append(np.array(obj.scale, dtype=float))

            if not args.skip_object_splitting:
                # Replace object if it is found in another category (train / test)
                if (scene_mode == 'train' and obj_id in test_obj_id_list) or (scene_mode == 'test' and obj_id in train_obj_id_list):
                    if obj_id in orig_swap_dict.keys():  # Use previously replaced object for duplicate instances
                        replace_obj_id = orig_swap_dict[obj_id]
                    else:
                        obj_cls = cls_label
                        obj_subcls = obj_id2subclass.get(obj_id, None)
                        if scene_mode == 'train':
                            if args.ignore_subclass:
                                cls_sample_list = train_cls2obj_id[obj_cls]
                            else:
                                if obj_subcls is not None and len(train_subcls2obj_id[obj_subcls]) >= SUBCLASS_MIN_COUNT:
                                    cls_sample_list = train_subcls2obj_id[obj_subcls]
                                else:
                                    cls_sample_list = train_cls2obj_id[obj_cls]

                            train_sample_tgt = [oid for oid in cls_sample_list if oid != obj_id]
                            train_aspect_vec = np.stack([obj_id2aspect_vec[oid] for oid in train_sample_tgt])
                            train_aspect_dist = np.linalg.norm(obj_id2aspect_vec[obj_id][None, :] - train_aspect_vec, axis=-1)
                            train_sample_range = np.argsort(train_aspect_dist)[:args.topk_aspect_sampling]
                            replace_obj_id = train_sample_tgt[np.random.choice(train_sample_range)]
                        else:
                            if args.ignore_subclass:
                                cls_sample_list = test_cls2obj_id[obj_cls]
                            else:
                                if obj_subcls is not None and len(test_subcls2obj_id[obj_subcls]) >= SUBCLASS_MIN_COUNT:
                                    cls_sample_list = test_subcls2obj_id[obj_subcls]
                                else:
                                    cls_sample_list = test_cls2obj_id[obj_cls]

                            test_sample_tgt = [oid for oid in cls_sample_list if oid != obj_id]
                            test_aspect_vec = np.stack([obj_id2aspect_vec[oid] for oid in test_sample_tgt])
                            test_aspect_dist = np.linalg.norm(obj_id2aspect_vec[obj_id][None, :] - test_aspect_vec, axis=-1)
                            test_sample_range = np.argsort(test_aspect_dist)[:args.topk_aspect_sampling]
                            replace_obj_id = test_sample_tgt[np.random.choice(test_sample_range)]
                        orig_swap_dict[obj_id] = replace_obj_id

                    replace_obj_model_path = os.path.join(args.obj_root, replace_obj_id, "raw_model.obj")

                    replace_obj_bbox_corners = obj_id2bbox_corner[replace_obj_id] * orig_scene_model["obj_scene_scales"][-1]
                    replace_obj_bbox_sizes = replace_obj_bbox_corners[0] - replace_obj_bbox_corners[1]
                    replace_obj_centroids = (replace_obj_bbox_corners[0] + replace_obj_bbox_corners[1]) / 2.

                    # Re-scale newly loaded object mesh
                    if args.rescaling_method == "optimal_scalar":
                        resize_rate = ((obj_bbox_sizes[0] * replace_obj_bbox_sizes[0]) + \
                            (obj_bbox_sizes[1] * replace_obj_bbox_sizes[1]) + \
                            (obj_bbox_sizes[2] * replace_obj_bbox_sizes[2])) / \
                            (replace_obj_bbox_sizes[0] ** 2 + replace_obj_bbox_sizes[1] ** 2 + replace_obj_bbox_sizes[2] ** 2)  # Minimizes squared error between bbox sizes
                    elif args.rescaling_method == "box_scale":
                        resize_rate = obj_bbox_sizes / replace_obj_bbox_sizes
                    else:
                        raise NotImplementedError("Other re-scaling methods not supported")

                    orig_scene_model["obj_scene_scales"][-1] *= resize_rate

                    replace_obj_bbox_corners = replace_obj_bbox_corners * resize_rate
                    replace_obj_bbox_sizes = replace_obj_bbox_corners[0] - replace_obj_bbox_corners[1]
                    replace_obj_centroids = (replace_obj_bbox_corners[0] + replace_obj_bbox_corners[1]) / 2.

                    orig_scene_model['obj_path'][-1] = replace_obj_model_path
                    orig_scene_model['obj_id'][-1] = replace_obj_id
                    orig_scene_model['bboxes'][-1] = replace_obj_bbox_sizes
                    orig_scene_model['bbox_centroids'][-1] = replace_obj_centroids

        orig_scene_model['bboxes'] = np.stack(orig_scene_model['bboxes'], axis=0)
        orig_scene_model['bbox_centroids'] = np.stack(orig_scene_model['bbox_centroids'], axis=0)
        orig_scene_model['trans'] = np.stack(orig_scene_model['trans'], axis=0)
        orig_scene_model['rot'] = np.stack(orig_scene_model['rot'], axis=0)
        orig_scene_model['obj_classes'] = np.array(orig_scene_model['obj_classes'])
        orig_scene_model['obj_classes_str'] = np.array(orig_scene_model['obj_classes_str'])
        orig_scene_model['obj_path'] = np.array(orig_scene_model['obj_path'])
        orig_scene_model['obj_id'] = np.array(orig_scene_model['obj_id'])
        orig_scene_model['obj_scene_scales'] = np.stack(orig_scene_model['obj_scene_scales'], axis=0).astype(float)

        # Load floorplan
        tr_floor = floor_plan_from_scene(scene, args.fp_texture_root)
        floorplans = np.asarray(tr_floor.vertices)
        floorplan_faces = np.asarray(tr_floor.faces)

        # Extract contour from 2D floorplan mesh
        floorplans = contour_from_floorplan_mesh(floorplans, floorplan_faces, simplify_contour=True)
        neg_floorplans = np.copy(floorplans)

        # Lines to use for wireframe creation
        floorplan_lines = np.stack([np.arange(floorplans.shape[0]), np.roll(np.arange(floorplans.shape[0]), shift=-1)], axis=1)
        neg_floorplan_lines = np.stack([np.arange(neg_floorplans.shape[0]), np.roll(np.arange(neg_floorplans.shape[0]), shift=-1)], axis=1)

        # Load walls to determine scene height
        wall_pts_list = []
        for ei in scene.extras:
            if "WallInner" in ei.model_type:
                render_mesh = ei.mesh_renderable(
                    offset=-scene.centroid,
                    colors=(0.8, 0.8, 0.8, 0.6)
                )

                tr_wall = [trimesh.Trimesh(
                    vertices=render_mesh.vertices,
                    vertex_colors=render_mesh._colors,
                    faces=ei.faces
                )]
                wall_pts_list.append(
                    np.asarray(tr_wall[0].vertices)
                )
        if len(wall_pts_list) != 0:
            wall_pts_np = np.concatenate(wall_pts_list, axis=0)
            scene_height = wall_pts_np[:, 1].max() - wall_pts_np[:, 1].min()
        else:
            scene_height = args.default_scene_height

        # Create 3D wireframes for floorplans
        if args.floorplan_type == "single_level_lines":
            floorplan_model = o3d.geometry.LineSet()
            floorplan_model.points = o3d.utility.Vector3dVector(floorplans)
            floorplan_model.lines = o3d.utility.Vector2iVector(floorplan_lines)

            neg_floorplan_model = o3d.geometry.LineSet()
            neg_floorplan_model.points = o3d.utility.Vector3dVector(neg_floorplans)
            neg_floorplan_model.lines = o3d.utility.Vector2iVector(neg_floorplan_lines)
        elif args.floorplan_type == "two_level_lines":
            # Generate 3D floorplan
            floorplan_down = np.copy(floorplans)
            floorplan_up = np.copy(floorplans)
            floorplan_up[:, 1] += scene_height
            floorplan_lines_down = floorplan_lines
            floorplan_lines_up = floorplan_lines + floorplan_down.shape[0]
            floorplan_lines_inter = np.concatenate([floorplan_lines_up[:, 0:1], floorplan_lines_down[:, 0:1]], axis=-1)

            floorplan_model = o3d.geometry.LineSet()
            floorplan_model.points = o3d.utility.Vector3dVector(np.concatenate([floorplan_down, floorplan_up], axis=0))
            floorplan_model.lines = o3d.utility.Vector2iVector(
                np.concatenate([floorplan_lines_down, floorplan_lines_inter, floorplan_lines_up], axis=0))

            neg_floorplan_down = np.copy(neg_floorplans)
            neg_floorplan_up = np.copy(neg_floorplans)
            neg_floorplan_up[:, 1] += scene_height
            neg_floorplan_lines_down = neg_floorplan_lines
            neg_floorplan_lines_up = neg_floorplan_lines + neg_floorplan_down.shape[0]
            neg_floorplan_lines_inter = np.concatenate([neg_floorplan_lines_up[:, 0:1], neg_floorplan_lines_down[:, 0:1]], axis=-1)

            neg_floorplan_model = o3d.geometry.LineSet()
            neg_floorplan_model.points = o3d.utility.Vector3dVector(np.concatenate([neg_floorplan_down, neg_floorplan_up], axis=0))
            neg_floorplan_model.lines = o3d.utility.Vector2iVector(
                np.concatenate([neg_floorplan_lines_down, neg_floorplan_lines_inter, neg_floorplan_lines_up], axis=0))
        elif args.floorplan_type == "walls":
            # Generate 3D floorplan
            floorplan_down = np.copy(floorplans)
            floorplan_up = np.copy(floorplans)
            floorplan_up[:, 1] += scene_height
            floorplan_lines_down = floorplan_lines
            floorplan_lines_up = floorplan_lines + floorplan_down.shape[0]
            floorplan_triangles_down = np.concatenate([floorplan_lines_up[:, 0:1], floorplan_lines_down], axis=-1)
            floorplan_triangles_down_flip = floorplan_triangles_down[:, [2, 1, 0]]
            floorplan_triangles_up = np.concatenate([floorplan_lines_down[:, 1:2], floorplan_lines_up], axis=-1)
            floorplan_triangles_up_flip = floorplan_triangles_up[:, [2, 1, 0]]
            floorplan_model = o3d.geometry.TriangleMesh()
            floorplan_model.vertices = o3d.utility.Vector3dVector(np.concatenate([floorplan_down, floorplan_up], axis=0))
            floorplan_model.triangles = o3d.utility.Vector3iVector(
                np.concatenate([floorplan_triangles_down, floorplan_triangles_up, floorplan_triangles_down_flip, floorplan_triangles_up_flip], axis=0))
            floorplan_model.paint_uniform_color((0., 0.5, 0.5))
            floorplan_model.compute_triangle_normals()
            floorplan_model.compute_vertex_normals()

            neg_floorplan_down = np.copy(neg_floorplans)
            neg_floorplan_up = np.copy(neg_floorplans)
            neg_floorplan_up[:, 1] += scene_height
            neg_floorplan_lines_down = neg_floorplan_lines
            neg_floorplan_lines_up = neg_floorplan_lines + neg_floorplan_down.shape[0]
            neg_floorplan_triangles_down = np.concatenate([neg_floorplan_lines_up[:, 0:1], neg_floorplan_lines_down], axis=-1)
            neg_floorplan_triangles_down_flip = neg_floorplan_triangles_down[:, [2, 1, 0]]
            neg_floorplan_triangles_up = np.concatenate([neg_floorplan_lines_down[:, 1:2], neg_floorplan_lines_up], axis=-1)
            neg_floorplan_triangles_up_flip = neg_floorplan_triangles_up[:, [2, 1, 0]]
            neg_floorplan_model = o3d.geometry.TriangleMesh()
            neg_floorplan_model.vertices = o3d.utility.Vector3dVector(np.concatenate([neg_floorplan_down, neg_floorplan_up], axis=0))
            neg_floorplan_model.triangles = o3d.utility.Vector3iVector(
                np.concatenate([neg_floorplan_triangles_down, neg_floorplan_triangles_up, neg_floorplan_triangles_down_flip, neg_floorplan_triangles_up_flip], axis=0))
            neg_floorplan_model.paint_uniform_color((0., 0.5, 0.5))
            neg_floorplan_model.compute_triangle_normals()
            neg_floorplan_model.compute_vertex_normals()
        else:
            raise NotImplementedError("Other floorplans not supported")

        # Initialize positive / negative / pair positive scene triplets
        pos_scene_model = copy.deepcopy(orig_scene_model)
        neg_scene_model = copy.deepcopy(orig_scene_model)
        pair_pos_scene_model = copy.deepcopy(orig_scene_model)
        vis_pos_scene_model_list = []
        vis_neg_scene_model_list = []
        vis_pair_pos_scene_model_list = []

        pos_scene_model['floorplan'] = floorplan_model
        neg_scene_model['floorplan'] = neg_floorplan_model
        pair_pos_scene_model['floorplan'] = floorplan_model
        vis_pos_scene_model_list.append({'name': 'pos_fp', 'geometry': floorplan_model})
        vis_neg_scene_model_list.append({'name': 'neg_fp', 'geometry': neg_floorplan_model})
        vis_pair_pos_scene_model_list.append({'name': 'pair_pos_fp', 'geometry': floorplan_model})

        if isinstance(floorplan_model, o3d.geometry.LineSet):
            vis_pos_scene_model_list.append({'name': 'pos_fp_kpts', 'geometry': keypoints_to_spheres(floorplan_model)})
            vis_neg_scene_model_list.append({'name': 'neg_fp_kpts', 'geometry': keypoints_to_spheres(neg_floorplan_model)})
            vis_pair_pos_scene_model_list.append({'name': 'pair_pos_fp_kpts', 'geometry': keypoints_to_spheres(floorplan_model)})

        num_scene_objects = len(scene.bboxes)

        # Generate objects
        pair_pos_swap_dict = {}
        for obj_idx, obj in enumerate(scene.bboxes):
            # Add noise to neg_model
            theta_noise = np.random.normal(loc=0., scale=np.deg2rad(args.max_rot_noise), size=(1, )).item()
            rot_noise = np.zeros((3, 3))
            rot_noise[0, 0] = np.cos(theta_noise)
            rot_noise[0, 2] = -np.sin(theta_noise)
            rot_noise[2, 0] = np.sin(theta_noise)
            rot_noise[2, 2] = np.cos(theta_noise)
            rot_noise[1, 1] = 1.

            trans_noise = np.zeros((1, 3))
            trans_noise[0, [0, 2]] = np.random.normal(loc=0., scale=args.max_trans_noise, size=(2, ))

            neg_scene_model['rot'][obj_idx] = rot_noise @ neg_scene_model['rot'][obj_idx]
            neg_scene_model['trans'][obj_idx] = neg_scene_model['trans'][obj_idx] + trans_noise

            # Replace object models to generation pair_pos_model
            if pair_pos_scene_model['obj_id'][obj_idx] in pair_pos_swap_dict.keys():  # Use previously replaced object for duplicate instances
                replace_obj_id = pair_pos_swap_dict[pair_pos_scene_model['obj_id'][obj_idx]]
            else:
                obj_cls = pair_pos_scene_model['obj_classes_str'][obj_idx]
                orig_obj_id = pair_pos_scene_model['obj_id'][obj_idx]
                obj_subcls = obj_id2subclass.get(orig_obj_id, None)
                if not args.skip_object_splitting:
                    if scene_mode == 'train':
                        if args.ignore_subclass:
                            cls_sample_list = train_cls2obj_id[obj_cls]
                        else:
                            if obj_subcls is not None and len(train_subcls2obj_id[obj_subcls]) >= SUBCLASS_MIN_COUNT:
                                cls_sample_list = train_subcls2obj_id[obj_subcls]
                            else:
                                cls_sample_list = train_cls2obj_id[obj_cls]

                        train_sample_tgt = [oid for oid in cls_sample_list if oid != orig_obj_id]
                        train_aspect_vec = np.stack([obj_id2aspect_vec[oid] for oid in train_sample_tgt])
                        train_aspect_dist = np.linalg.norm(obj_id2aspect_vec[orig_obj_id][None, :] - train_aspect_vec, axis=-1)
                        train_sample_range = np.argsort(train_aspect_dist)[:args.topk_aspect_sampling]
                        replace_obj_id = train_sample_tgt[np.random.choice(train_sample_range)]
                    else:
                        if args.ignore_subclass:
                            cls_sample_list = test_cls2obj_id[obj_cls]
                        else:
                            if obj_subcls is not None and len(test_subcls2obj_id[obj_subcls]) >= SUBCLASS_MIN_COUNT:
                                cls_sample_list = test_subcls2obj_id[obj_subcls]
                            else:
                                cls_sample_list = test_cls2obj_id[obj_cls]

                        test_sample_tgt = [oid for oid in cls_sample_list if oid != orig_obj_id]
                        test_aspect_vec = np.stack([obj_id2aspect_vec[oid] for oid in test_sample_tgt])
                        test_aspect_dist = np.linalg.norm(obj_id2aspect_vec[orig_obj_id][None, :] - test_aspect_vec, axis=-1)
                        test_sample_range = np.argsort(test_aspect_dist)[:args.topk_aspect_sampling]
                        replace_obj_id = test_sample_tgt[np.random.choice(test_sample_range)]
                else:
                    if args.ignore_subclass:
                        cls_sample_list = cls2obj_id[obj_cls]
                    else:
                        if obj_subcls is not None and len(subcls2obj_id[obj_subcls]) >= SUBCLASS_MIN_COUNT:
                            cls_sample_list = subcls2obj_id[obj_subcls]
                        else:
                            cls_sample_list = cls2obj_id[obj_cls]

                    sample_tgt = [oid for oid in cls_sample_list if oid != orig_obj_id]
                    aspect_vec = np.stack([obj_id2aspect_vec[oid] for oid in sample_tgt])
                    aspect_dist = np.linalg.norm(obj_id2aspect_vec[orig_obj_id][None, :] - aspect_vec, axis=-1)
                    sample_range = np.argsort(aspect_dist)[:args.topk_aspect_sampling]
                    replace_obj_id = sample_tgt[np.random.choice(sample_range)]
                pair_pos_swap_dict[pair_pos_scene_model['obj_id'][obj_idx]] = replace_obj_id

            replace_obj_model_path = os.path.join(args.obj_root, replace_obj_id, "raw_model.obj")

            replace_obj_bbox_corners = obj_id2bbox_corner[replace_obj_id] * pair_pos_scene_model["obj_scene_scales"][obj_idx]
            replace_obj_bbox_sizes = replace_obj_bbox_corners[0] - replace_obj_bbox_corners[1]
            replace_obj_centroids = (replace_obj_bbox_corners[0] + replace_obj_bbox_corners[1]) / 2.

            # Re-scale newly loaded object mesh
            pair_pos_obj_bbox_sizes = pair_pos_scene_model['bboxes'][obj_idx]
            if args.rescaling_method == "optimal_scalar":
                resize_rate = ((pair_pos_obj_bbox_sizes[0] * replace_obj_bbox_sizes[0]) + \
                    (pair_pos_obj_bbox_sizes[1] * replace_obj_bbox_sizes[1]) + \
                    (pair_pos_obj_bbox_sizes[2] * replace_obj_bbox_sizes[2])) / \
                    (replace_obj_bbox_sizes[0] ** 2 + replace_obj_bbox_sizes[1] ** 2 + replace_obj_bbox_sizes[2] ** 2)  # Minimizes squared error between bbox sizes
            elif args.rescaling_method == "box_scale":
                resize_rate = pair_pos_obj_bbox_sizes / replace_obj_bbox_sizes
            else:
                raise NotImplementedError("Other re-scaling methods not supported")

            pair_pos_scene_model["obj_scene_scales"][obj_idx] *= resize_rate

            replace_obj_bbox_corners = replace_obj_bbox_corners * resize_rate
            replace_obj_bbox_sizes = replace_obj_bbox_corners[0] - replace_obj_bbox_corners[1]
            replace_obj_centroids = (replace_obj_bbox_corners[0] + replace_obj_bbox_corners[1]) / 2.

            pair_pos_scene_model['obj_path'][obj_idx] = replace_obj_model_path
            pair_pos_scene_model['obj_id'][obj_idx] = replace_obj_id
            pair_pos_scene_model['bboxes'][obj_idx] = replace_obj_bbox_sizes
            pair_pos_scene_model['bbox_centroids'][obj_idx] = replace_obj_centroids

            if args.visualize_scene:
                scene_model_list = [pos_scene_model, neg_scene_model, pair_pos_scene_model]

                for s_idx, scene_model in enumerate(scene_model_list):
                    tr_mesh = trimesh_load_with_postprocess(scene_model['obj_path'][obj_idx], postprocess_type='bottom_crop')
                    texture_path = scene_model['obj_path'][obj_idx].replace("raw_model.obj", "texture.png")
                    if not os.path.exists(texture_path):
                        texture_path = scene_model['obj_path'][obj_idx].replace("raw_model.obj", "texture.jpg")

                    tr_mesh.visual.material.image = Image.open(texture_path)
                    tr_mesh.vertices *= scene_model['obj_scene_scales'][obj_idx]
                    tr_mesh.vertices[...] = \
                        tr_mesh.vertices.dot(scene_model['rot'][obj_idx].T) + scene_model['trans'][obj_idx]

                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(tr_mesh.vertices)
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(tr_mesh.faces)
                    uvs = tr_mesh.visual.uv
                    triangles_uvs = []
                    for i in range(3):
                        triangles_uvs.append(uvs[tr_mesh.faces[:, i]].reshape(-1, 1, 2))
                    triangles_uvs = np.concatenate(triangles_uvs, axis=1).reshape(-1, 2)

                    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
                    o3d_mesh.textures = [o3d.geometry.Image(np.asarray(tr_mesh.visual.material.image))]
                    o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(tr_mesh.faces))

                    material = rendering.MaterialRecord()
                    material.shader = 'defaultUnlit'
                    material.albedo_img = o3d.geometry.Image(np.asarray(tr_mesh.visual.material.image))

                    if s_idx == 0:  # Positive
                        vis_pos_scene_model_list.append({'name': f'pos_mesh_{obj_idx}', 'geometry': o3d_mesh, 'material': material})
                    elif s_idx == 1:  # Negative
                        vis_neg_scene_model_list.append({'name': f'neg_mesh_{obj_idx}', 'geometry': o3d_mesh, 'material': material})
                    else:  # Pair-positive
                        vis_pair_pos_scene_model_list.append({'name': f'pair_pos_mesh_{obj_idx}', 'geometry': o3d_mesh, 'material': material})

        # Optionally visualize scene
        if args.visualize_scene:
            o3d.visualization.draw(vis_pos_scene_model_list, show_skybox=False)
            o3d.visualization.draw(vis_neg_scene_model_list, show_skybox=False)
            o3d.visualization.draw(vis_pair_pos_scene_model_list, show_skybox=False)

        # Save generated scenes
        if not args.skip_object_splitting:
            if scene_mode == 'train':
                curr_scene_save_root = os.path.join(args.save_root, f"{scene_mode}_scene_{scene_idx}")
            else:
                curr_scene_save_root = os.path.join(args.save_root, f"{scene_mode}_scene_{scene_idx - num_train_scenes}")
        else:
            curr_scene_save_root = os.path.join(args.save_root, f"scene_{scene_idx}")
        os.makedirs(curr_scene_save_root, exist_ok=True)

        # Load point & connectivity information from scene models
        curr_scene_path = os.path.join(curr_scene_save_root, "scene.npz")
        scene_model_list = [pos_scene_model, pair_pos_scene_model, neg_scene_model]
        scene_names_list = ['pos', 'pair_pos', 'neg']
        save_scene_dict = {}
        is_valid = True

        for scene_name, scene_model in zip(scene_names_list, scene_model_list):
            scene_num_obj = len(scene_model['obj_id'])
            if isinstance(scene_model['floorplan'], o3d.geometry.TriangleMesh):  # Floorplan is saved as triange mesh
                scene_fp_points = np.asarray(scene_model['floorplan'].vertices)
                scene_fp_lines = np.asarray(floorplan_lines)
            else:  # Floorplan is saved as line set
                scene_fp_points = np.asarray(scene_model['floorplan'].points)
                scene_fp_lines = np.asarray(scene_model['floorplan'].lines)

            # Check validity before saving
            if np.isnan(scene_model['trans']).sum() != 0:
                is_valid = False
                break
            if np.isnan(scene_model['rot']).sum() != 0:
                is_valid = False
                break
            if np.isnan(scene_model['obj_scene_scales']).sum() != 0:
                is_valid = False
                break
            if len(scene_fp_points) == 0:
                is_valid = False
                break

            save_scene_dict[scene_name + "_fp_points"] = scene_fp_points
            save_scene_dict[scene_name + "_fp_lines"] = scene_fp_lines
            save_scene_dict[scene_name + "_bboxes"] = scene_model['bboxes']
            save_scene_dict[scene_name + "_trans"] = scene_model['trans']
            save_scene_dict[scene_name + "_rot"] = scene_model['rot']
            save_scene_dict[scene_name + "_bbox_centroids"] = scene_model['bbox_centroids']
            save_scene_dict[scene_name + "_obj_path"] = scene_model['obj_path']
            save_scene_dict[scene_name + "_obj_classes"] = scene_model['obj_classes']
            save_scene_dict[scene_name + "_obj_classes_str"] = scene_model['obj_classes_str']
            save_scene_dict[scene_name + "_obj_subclasses_str"] = scene_model['obj_subclasses_str']
            save_scene_dict[scene_name + "_obj_scene_scales"] = scene_model['obj_scene_scales']
            save_scene_dict[scene_name + "_obj_id"] = scene_model['obj_id']

        if is_valid:
            np.savez_compressed(curr_scene_path, **save_scene_dict)
        else:
            subprocess.run(['rm', '-rf', curr_scene_save_root])
