import argparse
import torch
import numpy as np
import random
import os
from glob import glob
from tqdm import trange
import open3d as o3d
import copy
from arkit.utils import (
    RollingSampler,
    keypoints_to_spheres,
    str2cls_arkit
)
from tqdm import tqdm
import pickle
from arkit.invalid_files import PROBLEMATIC_SCENE_ID
import subprocess

SUBCLASS_MIN_COUNT = 5  # Minimum number of objects in subclasses to be considered usable


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_root", default="./data/arkit_scenes_processed/objects/", type=str)
    parser.add_argument("--scene_root", help="Root path to .npz files specifying scenes for inspection", default="./data/arkit_scenes_processed/scenes/", type=str)
    parser.add_argument("--raw_scene_root", help="Root path to folders containing .ply and .json files for raw scene data", default="./data/arkit_scenes/", type=str)
    parser.add_argument("--save_root", help="Directory for saving generated scenes", default="./data/arkit_scenes_gen/", type=str)
    parser.add_argument("--aspect_ratio_path", help="Path to saving object aspect ratios", default="./data/arkit_scenes_gen_assets/obj_aspect_ratio.pkl", type=str)
    parser.add_argument("--bbox_corner_path", help="Path to saving bounding box corners", default="./data/arkit_scenes_gen_assets/obj_bbox_corner.pkl", type=str)
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)
    parser.add_argument("--max_trans_noise", help="Maximum noise level for translation", default=0.1, type=float)
    parser.add_argument("--max_rot_noise", help="Maximum noise level for rotation in degrees", default=22.5, type=float)
    parser.add_argument("--max_fp_noise", help="Maximum noise level for floorplan perturbation", default=0.2, type=float)
    parser.add_argument("--visualize_scene", help="Optionally visualize scene", action="store_true")
    parser.add_argument("--rescaling_method", type=str, help="Type of re-scaling to perform on replaced objects", default="box_scale")
    parser.add_argument("--topk_aspect_sampling", help="Top-k objects according to aspect ratio distance to sample from", type=int, default=100)
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
    train_obj_dirs = sorted(glob(os.path.join(args.obj_root, "train", "*.ply")))
    test_obj_dirs = sorted(glob(os.path.join(args.obj_root, "test", "*.ply")))
    obj_dirs = train_obj_dirs + test_obj_dirs

    num_train_objs = len(train_obj_dirs)
    num_test_objs = len(test_obj_dirs)
    num_objs = num_train_objs + num_test_objs

    obj_classes = [
        "cabinet",  # 0
        "refrigerator",  # 1
        "shelf",  # 2
        "stove",  # 3
        "bed",  # 4
        "sink",  # 5
        "washer",  # 6
        "toilet",  # 7
        "bathtub",  # 8
        "oven",  # 9
        "dishwasher",  # 10
        "fireplace",  # 11
        "stool",  # 12
        "chair",  # 13
        "table",  # 14
        "tv_monitor",  # 15
        "sofa"  # 16
    ]
    cls2obj_id = {}
    obj_id2class = {}

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

    for obj_path in tqdm(obj_dirs, desc="Parsing object info"):
        obj_filename = os.path.basename(obj_path)
        obj_id = obj_filename.replace('.ply', '')
        obj_class = obj_id.replace("_" + obj_id.split('_')[-1], '')

        if obj_class not in cls2obj_id.keys():
            cls2obj_id[obj_class] = []
        cls2obj_id[obj_class].append(obj_id)
        obj_id2class[obj_id] = obj_class

        if caching_aspect_ratio or caching_bbox_corner:
            # Compute aspect ratio
            obj_mesh = o3d.io.read_triangle_mesh(obj_path)
            bbox_size = np.asarray(obj_mesh.vertices).max(0) - np.asarray(obj_mesh.vertices).min(0)
            aspect_vec = bbox_size / bbox_size[0]
            bbox_corners = np.stack([
                np.asarray(obj_mesh.vertices).max(0),
                np.asarray(obj_mesh.vertices).min(0)
            ], axis=0)  # (2, 3)
            if caching_aspect_ratio:
                obj_id2aspect_vec[obj_id] = aspect_vec
            if caching_bbox_corner:
                obj_id2bbox_corner[obj_id] = bbox_corners

    # Cache obj_id2aspect_vec as this takes quite a while to compute
    if caching_aspect_ratio:
        with open(args.aspect_ratio_path, 'wb') as f:
            pickle.dump(obj_id2aspect_vec, f)

    # Cache obj_id2bbox_corner as this takes quite a while to compute
    if caching_bbox_corner:
        with open(args.bbox_corner_path, 'wb') as f:
            pickle.dump(obj_id2bbox_corner, f)

    # Parse rooms in ARKIT
    arkit_rooms = sorted(glob(os.path.join(args.scene_root, "*/*.npz")))
    arkit_rooms = [room for room in arkit_rooms if room.strip('.npz') not in PROBLEMATIC_SCENE_ID]

    print(f"Number of rooms: {len(arkit_rooms)}")

    # Identify scenes and objects to use for training
    test_scenes = [room for room in arkit_rooms if 'test' in room]
    train_scenes = [room for room in arkit_rooms if 'train' in room]

    num_test_scenes = len(test_scenes)
    num_train_scenes = len(train_scenes)
    num_scenes = num_train_scenes + num_test_scenes  # NOTE: We fix the number of scenes parameter for ARKIT scenes as it is provided by the dataset

    test_scene_sampler = RollingSampler(test_scenes)
    train_scene_sampler = RollingSampler(train_scenes)

    obj_id_list = [os.path.basename(obj_path).replace('.ply', '') for obj_path in obj_dirs]
    test_obj_id_list = [os.path.basename(obj_path).replace('.ply', '') for obj_path in obj_dirs if 'test' in obj_path]
    train_obj_id_list = [os.path.basename(obj_path).replace('.ply', '') for obj_path in obj_dirs if 'train' in obj_path]

    # Make separate object mappings for train and test
    train_cls2obj_id = {}
    test_cls2obj_id = {}
    for cls_name in cls2obj_id.keys():
        train_cls2obj_id[cls_name] = []
        test_cls2obj_id[cls_name] = []
        for obj_id in cls2obj_id[cls_name]:
            if obj_id in test_obj_id_list:
                test_cls2obj_id[cls_name].append(obj_id)
            else:
                train_cls2obj_id[cls_name].append(obj_id)

    # Note the scene scales follow the object point cloud scales
    for scene_idx in trange(num_scenes):
        if scene_idx < num_train_scenes:
            scene_mode = 'train'
        else:
            scene_mode = 'test'

        # Scenes will be generated in a roullette-based fashion
        if scene_mode == 'train':
            scene_path = train_scene_sampler.sample(1)[0]
        else:
            scene_path = test_scene_sampler.sample(1)[0]

        scene = np.load(scene_path)
        num_scene_objects = len(scene['obj_ids'])

        # Load objects from original scene
        orig_scene_model = {
            'floorplan': None,
            'bboxes': [],  # Bboxes are at canonical pose
            'trans': [],
            'rot': [],
            'bbox_centroids': [],  # Centroid coordinates of canonical object model bounding boxes
            'obj_classes': [],
            'obj_classes_str': [],
            'obj_path': [],
            'obj_id': [],
            'obj_scene_scales': [],  # Scale values of objects within each scene (all ones for ARKIT)
            'orig_scene_path': []  # Scene ID of original scene used for generating the sample
        }

        # Parse original scene path
        scene_id = os.path.basename(scene_path).replace(".npz", "")
        orig_scene_path = os.path.join(args.raw_scene_root, scene_mode, scene_id)
        orig_scene_model['orig_scene_path'].append(orig_scene_path)

        for obj_idx in range(num_scene_objects):
            obj_id = scene['obj_ids'][obj_idx]
            obj_model_path = os.path.join(args.obj_root, scene_mode, f"{obj_id}.ply")

            obj_bbox_corners = obj_id2bbox_corner[obj_id]
            obj_bbox_sizes = obj_bbox_corners[0] - obj_bbox_corners[1]
            obj_centroids = (obj_bbox_corners[0] + obj_bbox_corners[1]) / 2.
            rot_mtx = scene['rot'][obj_idx]
            trans_mtx = scene['trans'][obj_idx]

            orig_scene_model['bbox_centroids'].append(obj_centroids)
            orig_scene_model['bboxes'].append(obj_bbox_sizes)
            orig_scene_model['trans'].append(trans_mtx)
            orig_scene_model['rot'].append(rot_mtx)

            # Load object classes
            cls_label = obj_id2class[obj_id]
            cls_label_id = str2cls_arkit(cls_label, return_type='int')
            orig_scene_model['obj_classes'].append(cls_label_id)
            orig_scene_model['obj_classes_str'].append(cls_label)
            orig_scene_model['obj_path'].append(obj_model_path)
            orig_scene_model['obj_id'].append(obj_id)
            orig_scene_model['obj_scene_scales'].append(np.ones([3, ], dtype=float))

        orig_scene_model['bboxes'] = np.stack(orig_scene_model['bboxes'], axis=0)
        orig_scene_model['bbox_centroids'] = np.stack(orig_scene_model['bbox_centroids'], axis=0)
        orig_scene_model['trans'] = np.stack(orig_scene_model['trans'], axis=0)
        orig_scene_model['rot'] = np.stack(orig_scene_model['rot'], axis=0)
        orig_scene_model['obj_classes'] = np.array(orig_scene_model['obj_classes'])
        orig_scene_model['obj_classes_str'] = np.array(orig_scene_model['obj_classes_str'])
        orig_scene_model['obj_path'] = np.array(orig_scene_model['obj_path'])
        orig_scene_model['obj_id'] = np.array(orig_scene_model['obj_id'])
        orig_scene_model['obj_scene_scales'] = np.stack(orig_scene_model['obj_scene_scales'], axis=0).astype(float)
        orig_scene_model['orig_scene_path'] = np.array(orig_scene_model['orig_scene_path'])

        # Load floorplan
        floorplans = scene['fp_points']
        floorplan_lines = scene['fp_lines']

        floorplan_model = o3d.geometry.LineSet()
        floorplan_model.points = o3d.utility.Vector3dVector(floorplans)
        floorplan_model.lines = o3d.utility.Vector2iVector(floorplan_lines)

        neg_floorplan_model = o3d.geometry.LineSet()
        neg_floorplan_model.points = o3d.utility.Vector3dVector(floorplans)
        neg_floorplan_model.lines = o3d.utility.Vector2iVector(floorplan_lines)

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

        vis_pos_scene_model_list.append({'name': 'pos_fp_kpts', 'geometry': keypoints_to_spheres(floorplan_model)})
        vis_neg_scene_model_list.append({'name': 'neg_fp_kpts', 'geometry': keypoints_to_spheres(neg_floorplan_model)})
        vis_pair_pos_scene_model_list.append({'name': 'pair_pos_fp_kpts', 'geometry': keypoints_to_spheres(floorplan_model)})

        # Generate object swaps
        pair_pos_swap_dict = {}
        for obj_idx in range(num_scene_objects):
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

                if scene_mode == 'train':
                    cls_sample_list = train_cls2obj_id[obj_cls]
                    train_sample_tgt = [oid for oid in cls_sample_list if oid != orig_obj_id]
                    train_aspect_vec = np.stack([obj_id2aspect_vec[oid] for oid in train_sample_tgt])
                    train_aspect_dist = np.linalg.norm(obj_id2aspect_vec[orig_obj_id][None, :] - train_aspect_vec, axis=-1)
                    train_sample_range = np.argsort(train_aspect_dist)[:args.topk_aspect_sampling]
                    replace_obj_id = train_sample_tgt[np.random.choice(train_sample_range)]
                else:
                    cls_sample_list = test_cls2obj_id[obj_cls]
                    test_sample_tgt = [oid for oid in cls_sample_list if oid != orig_obj_id]
                    test_aspect_vec = np.stack([obj_id2aspect_vec[oid] for oid in test_sample_tgt])
                    test_aspect_dist = np.linalg.norm(obj_id2aspect_vec[orig_obj_id][None, :] - test_aspect_vec, axis=-1)
                    test_sample_range = np.argsort(test_aspect_dist)[:args.topk_aspect_sampling]
                    replace_obj_id = test_sample_tgt[np.random.choice(test_sample_range)]
                pair_pos_swap_dict[pair_pos_scene_model['obj_id'][obj_idx]] = replace_obj_id

            replace_obj_model_path = os.path.join(args.obj_root, scene_mode, f"{replace_obj_id}.ply")

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
                    o3d_mesh = o3d.io.read_triangle_mesh(scene_model['obj_path'][obj_idx])
                    canonical_vertices = np.asarray(o3d_mesh.vertices)
                    transformed_vertices = canonical_vertices * scene_model['obj_scene_scales'][obj_idx].reshape(-1, 3)
                    transformed_vertices = transformed_vertices @ scene_model['rot'][obj_idx].T + scene_model['trans'][obj_idx: obj_idx + 1]
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)

                    if s_idx == 0:  # Positive
                        vis_pos_scene_model_list.append({'name': f'pos_mesh_{obj_idx}', 'geometry': o3d_mesh})
                    elif s_idx == 1:  # Negative
                        vis_neg_scene_model_list.append({'name': f'neg_mesh_{obj_idx}', 'geometry': o3d_mesh})
                    else:  # Pair-positive
                        vis_pair_pos_scene_model_list.append({'name': f'pair_pos_mesh_{obj_idx}', 'geometry': o3d_mesh})

        # Optionally visualize scene
        if args.visualize_scene:
            o3d.visualization.draw(vis_pos_scene_model_list, show_skybox=False)
            o3d.visualization.draw(vis_neg_scene_model_list, show_skybox=False)
            o3d.visualization.draw(vis_pair_pos_scene_model_list, show_skybox=False)

        # Save generated scenes
        if scene_mode == 'train':
            curr_scene_save_root = os.path.join(args.save_root, f"{scene_mode}_scene_{scene_idx}")
        else:
            curr_scene_save_root = os.path.join(args.save_root, f"{scene_mode}_scene_{scene_idx - num_train_scenes}")
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
            save_scene_dict[scene_name + "_obj_scene_scales"] = scene_model['obj_scene_scales']
            save_scene_dict[scene_name + "_obj_id"] = scene_model['obj_id']
            save_scene_dict[scene_name + "_orig_scene_path"] = scene_model['orig_scene_path']

        if is_valid:
            np.savez_compressed(curr_scene_path, **save_scene_dict)
        else:
            subprocess.run(['rm', '-rf', curr_scene_save_root])
