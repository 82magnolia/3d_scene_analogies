import argparse
import torch
import numpy as np
import random
import os
from glob import glob
from tqdm import trange
import open3d as o3d
from arkit.utils import (
    choice_without_replacement,
    generate_yaw_points,
    yaw2rot_mtx,
    CollisionFreePlacer,
    o3d_geometry_list_shift
)
from tqdm import tqdm
import pickle
import subprocess
from arkit.data_utils import build_dense_3d_scene, generate_obj_surface_points
from matplotlib import colormaps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_root", default="./data/arkit_scenes_processed/objects/", type=str)
    parser.add_argument("--orig_scene_root", help="Root path to .npz files specifying scenes to augment", type=str)
    parser.add_argument("--obj_model_info_path", help="Path to .json file specifying object for inspection", default="./data/3D-FUTURE-model/model_info.json", type=str)
    parser.add_argument("--save_root", help="Directory for saving generated scnees", default=None, type=str)
    parser.add_argument("--aspect_ratio_path", help="Path to saving object aspect ratios", default="./data/3d_front_gen_assets/obj_aspect_ratio.pkl", type=str)
    parser.add_argument("--bbox_corner_path", help="Path to saving bounding box corners", default="./data/3d_front_gen_assets/obj_bbox_corner.pkl", type=str)
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)
    parser.add_argument("--global_num_rot", help="Number of yaw angle splits to make for global rotation perturbation", default=4, type=int)
    parser.add_argument("--fp_max_perturb_rate", help="Maximum perturbation rate for floorplan points", default=0.4, type=float)
    parser.add_argument("--fp_max_noise", help="Maximum noise level for floorplan points", default=0.1, type=float)
    parser.add_argument("--num_group_keep_range", help="Range of number of objects to keep per scene", default=[2, 4], type=int, nargs=2)
    parser.add_argument("--num_pose_perturb_rate", help="Rate of initially removed objects to be added back for pose perturbation", default=0.5, type=float)
    parser.add_argument("--max_trans_noise", help="Maximum noise level for translation", default=0.05, type=float)
    parser.add_argument("--max_rot_noise", help="Maximum noise level for rotation in degrees", default=10., type=float)
    parser.add_argument("--num_match_obj_add_range", help="Range of number of objects to add per scene using scene histogram matching", default=[2, 5], nargs=2, type=int)
    parser.add_argument("--num_match_place_points", help="Number of initial placement hypotheses to adding objects from matched scnees", default=1000, type=int)
    parser.add_argument("--visualize_scene", help="Optionally visualize scene", action="store_true")
    parser.add_argument("--visualize_gt_match", help="Optionally visualize ground-truth matches", action="store_true")
    parser.add_argument("--num_classes", help="Number of semantics classes in objects", default=17, type=int)
    parser.add_argument("--point_sample_root", help="Root folder containing point samples from object mesh files (currently used for scene generation without feature extraction)", default="./data/3d_future_point_samples/")
    args = parser.parse_args()

    """
    NOTE: This script takes an existing scene dataset as input and augments it into a new scene containing a selected group of objects.

    Augmentation is performed through a four step process.
        1. For each original scene apply floorplan perturbation and random global rotation. Also choose object group to keep.
        2. For object not kept, remove them or apply pose perturbation. (Make match_obj_idx field need for loading)
        3. Sample a scene with similar object semantics histograms.
        4. For object removed, randomly add objects from the sampled scene. Use rotations from sampled scene and translations from empty space sampling.
    """

    # Fix seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.save_root is None:
        save_root = args.orig_scene_root.strip('/') + '_augment/'
    else:
        save_root = args.save_root + '/'

    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Parse repository of object shapes
    train_obj_dirs = sorted(glob(os.path.join(args.obj_root, "train", "*.ply")))
    test_obj_dirs = sorted(glob(os.path.join(args.obj_root, "test", "*.ply")))
    obj_dirs = train_obj_dirs + test_obj_dirs

    num_train_objs = len(train_obj_dirs)
    num_test_objs = len(test_obj_dirs)
    num_objs = num_train_objs + num_test_objs

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

    for obj_path in tqdm(obj_dirs, desc="Parsing object info", total=len(obj_dirs)):
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

    # Parse rooms in original dataset
    orig_scene_files = sorted(glob(os.path.join(args.orig_scene_root, "**", "*.npz")))
    save_scene_files = [s.replace(args.orig_scene_root, save_root) for s in orig_scene_files]

    train_scene_files = [s for s in orig_scene_files if 'train' in s]
    test_scene_files = [s for s in orig_scene_files if 'test' in s]
    print(f"Number of rooms: {len(orig_scene_files)}")

    # NOTE: The number of scenes can be larger than the actual number of scenes in 3D-FRONT
    num_scenes = len(orig_scene_files)

    # Compute object histograms for scenes
    train_semantic_hist = np.zeros([len(train_scene_files), args.num_classes])
    for path_idx, path in enumerate(train_scene_files):
        scene_info = np.load(path)
        hist_vec = np.bincount(scene_info['pos_obj_classes'], minlength=args.num_classes).astype(float)  # / scene_info['pos_obj_classes'].shape[0]
        train_semantic_hist[path_idx] = hist_vec

    test_semantic_hist = np.zeros([len(test_scene_files), args.num_classes])
    for path_idx, path in enumerate(test_scene_files):
        scene_info = np.load(path)
        hist_vec = np.bincount(scene_info['pos_obj_classes'], minlength=args.num_classes).astype(float)  # / scene_info['pos_obj_classes'].shape[0]
        test_semantic_hist[path_idx] = hist_vec

    # Note the scene scales follow the object point cloud scales
    train_scene_idx = -1  # Index values used for keeping track of train scenes
    test_scene_idx = -1  # Index values used for keeping track of test scenes
    for scene_idx in trange(num_scenes):
        if 'train' in orig_scene_files[scene_idx]:
            scene_type = 'train'
            train_scene_idx += 1
        else:
            scene_type = 'test'
            test_scene_idx += 1

        # Load scene and apply floorplan + rotation perturbation
        scene_path = orig_scene_files[scene_idx]
        scene_triplet = np.load(scene_path)
        augment_scene_triplet = {k: v for (k, v) in scene_triplet.items() if 'pair_pos' in k}  # We only augment scenes labelled as 'pair_pos'

        # Floorplan perturbation (NOTE: perturbation is done in pairs to ensure segments are moved in parallel)
        fp_points = augment_scene_triplet['pair_pos_fp_points']
        fp_points_lower = fp_points[:fp_points.shape[0] // 2]
        num_corners = fp_points_lower.shape[0]
        fp_perturb_num = int(num_corners * args.fp_max_perturb_rate)
        fp_perturb_idx_list = choice_without_replacement(list(range(num_corners)), fp_perturb_num)

        for p_idx in fp_perturb_idx_list:
            fp_0 = fp_points_lower[p_idx % num_corners]
            fp_1 = fp_points_lower[(p_idx + 1) % num_corners]
            noise_val = (2 * random.random() - 1) * args.fp_max_noise

            dir_vec = (fp_0 - fp_1) / np.linalg.norm(fp_0 - fp_1, axis=-1)
            norm_vec = dir_vec @ np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]).T
            fp_0 += noise_val * norm_vec
            fp_1 += noise_val * norm_vec

            fp_points_lower[p_idx % num_corners] = fp_0
            fp_points_lower[(p_idx + 1) % num_corners] = fp_1

        fp_points[fp_points.shape[0] // 2:, [0, 2]] = fp_points_lower[:, [0, 2]]
        augment_scene_triplet['pair_pos_fp_points'] = fp_points

        # Global rotation perturbation
        yaw_points = generate_yaw_points(args.global_num_rot)
        rot_matrices = yaw2rot_mtx(yaw_points, apply_xz_flip=True).float().numpy()
        global_rot = rot_matrices[random.choice(range(rot_matrices.shape[0]))]

        # NOTE: We assume a transformation of X' = X @ R.T + T
        augment_scene_triplet['pair_pos_fp_points'] = augment_scene_triplet['pair_pos_fp_points'] @ global_rot.T
        augment_scene_triplet['pair_pos_trans'] = augment_scene_triplet['pair_pos_trans'] @ global_rot.T
        augment_scene_triplet['pair_pos_rot'] = global_rot[None, ...] @ augment_scene_triplet['pair_pos_rot']
        augment_scene_triplet['pair_pos_global_rot'] = global_rot  # Global rotation transformation saved for dense scene visualization using original scene mesh

        global_augment_scene_triplet = {k: np.copy(v) for k, v in augment_scene_triplet.items()}  # Globally augmented scene cached for future use in pose perturbation

        # Choose object group to keep
        obj_centroids = augment_scene_triplet['pair_pos_trans']  # Compute centroid distances and make groups from top-K distances
        obj_dist_mtx = np.linalg.norm(obj_centroids[:, None, :] - obj_centroids[None, :, :], axis=-1)
        obj_dist_mtx[np.diag_indices(obj_dist_mtx.shape[0])] = np.inf

        # Choose seed object and select top-K neighbors
        orig_num_scene_objects = augment_scene_triplet['pair_pos_trans'].shape[0]
        obj_inst_labels = np.arange(orig_num_scene_objects)
        topk = min(random.randint(*args.num_group_keep_range) - 1, orig_num_scene_objects - 1)
        keep_count = topk + 1
        sampling_prob = augment_scene_triplet['pair_pos_bboxes'].mean(axis=-1) / augment_scene_triplet['pair_pos_bboxes'].mean(axis=-1).sum()
        seed_obj_idx = np.random.choice(obj_inst_labels, p=sampling_prob)  # Bias seed object sampling towards choosing large objects
        topk_idx = np.argpartition(obj_dist_mtx, kth=topk, axis=-1)[seed_obj_idx, :topk]
        keep_obj_idx_list = [obj_inst_labels[seed_obj_idx]] + obj_inst_labels[topk_idx].tolist()
        reject_obj_idx_list = [obj_idx for obj_idx in obj_inst_labels if obj_idx not in keep_obj_idx_list]

        # Drop objects not included and make an index map
        augment_scene_triplet['pair_pos_idx_map'] = {orig_idx: keep_obj_idx for orig_idx, keep_obj_idx in enumerate(keep_obj_idx_list)}

        for k in augment_scene_triplet.keys():
            if k not in ['pair_pos_idx_map', 'pair_pos_fp_points', 'pair_pos_fp_lines', 'pair_pos_orig_scene_path', 'pair_pos_global_rot']:
                augment_scene_triplet[k] = augment_scene_triplet[k][keep_obj_idx_list]

        # Randomly add rejected objects back with random pose perturbation
        pose_perturb_count = int(len(reject_obj_idx_list) * args.num_pose_perturb_rate)
        pose_perturb_obj_idx_list = choice_without_replacement(reject_obj_idx_list, pose_perturb_count)
        pose_perturb_scene_triplet = {k: np.copy(v[pose_perturb_obj_idx_list]) for k, v in global_augment_scene_triplet.items() if k not in ['pair_pos_idx_map', 'pair_pos_fp_points', 'pair_pos_fp_lines', 'pair_pos_orig_scene_path', 'pair_pos_global_rot']}
        for obj_idx, pose_perturb_obj_idx in enumerate(pose_perturb_obj_idx_list):
            augment_scene_triplet['pair_pos_idx_map'][obj_idx + keep_count] = -1
            noise_trans = (2 * np.random.rand(2) - 1) * args.max_trans_noise  # Only add xz-translation
            pose_perturb_scene_triplet['pair_pos_trans'][obj_idx, [0, 2]] = global_augment_scene_triplet['pair_pos_trans'][pose_perturb_obj_idx, [0, 2]] + noise_trans
            pose_perturb_scene_triplet['pair_pos_trans'][obj_idx, 1] = global_augment_scene_triplet['pair_pos_trans'][pose_perturb_obj_idx, 1]

            noise_yaw = (2 * np.random.rand(1, ) - 1) * np.deg2rad(args.max_rot_noise)  # Only add y-rotation
            noise_rot = yaw2rot_mtx(torch.from_numpy(noise_yaw), apply_xz_flip=False).float().numpy()[0]
            pose_perturb_scene_triplet['pair_pos_rot'][obj_idx] = noise_rot @ global_augment_scene_triplet['pair_pos_rot'][pose_perturb_obj_idx]

        for k in augment_scene_triplet.keys():
            if k not in ['pair_pos_idx_map', 'pair_pos_fp_points', 'pair_pos_fp_lines', 'pair_pos_orig_scene_path', 'pair_pos_global_rot']:
                augment_scene_triplet[k] = np.concatenate([augment_scene_triplet[k], pose_perturb_scene_triplet[k]], axis=0)

        # Sample scene with similar semantic histograms
        if scene_type == 'train':
            search_semantic_hist = train_semantic_hist
            search_scene_files = train_scene_files
            curr_hist_vec = search_semantic_hist[train_scene_idx: train_scene_idx + 1]
        else:
            search_semantic_hist = test_semantic_hist
            search_scene_files = test_scene_files
            curr_hist_vec = search_semantic_hist[test_scene_idx: test_scene_idx + 1]

        hist_dist = np.linalg.norm(curr_hist_vec - search_semantic_hist, axis=-1)
        hist_dist[train_scene_idx] = np.inf
        match_path_idx = hist_dist.argmin()
        match_scene_triplet = np.load(search_scene_files[match_path_idx])
        match_scene_triplet = {k: v for (k, v) in match_scene_triplet.items() if 'pair_pos' in k}  # We only augment scenes labelled as 'pair_pos'

        match_obj_add_count = random.randint(*args.num_match_obj_add_range)
        match_num_scene_objects = match_scene_triplet['pair_pos_trans'].shape[0]
        match_obj_inst_labels = np.arange(match_num_scene_objects)

        # Incrementally place objects while avoiding collisions
        augment_obj_radius = np.linalg.norm(augment_scene_triplet['pair_pos_bboxes'][:, [0, 2]] / 2, axis=-1)
        match_obj_radius = np.linalg.norm(match_scene_triplet['pair_pos_bboxes'][:, [0, 2]] / 2, axis=-1)
        curr_obj_add_count = 0
        match_add_triplet = {k: [] for k in match_scene_triplet.keys() if 'pair_pos' in k}
        match_add_triplet['pair_pos_idx_map'] = {}
        augment_obj_centroids = (augment_scene_triplet['pair_pos_bbox_centroids'].reshape(-1, 1, 3) @ np.transpose(augment_scene_triplet['pair_pos_rot'], (0, 2, 1)) + \
            augment_scene_triplet['pair_pos_trans'].reshape(-1, 1, 3)).reshape(-1, 3)

        placer = CollisionFreePlacer(augment_scene_triplet['pair_pos_fp_points'], args.num_match_place_points, augment_obj_centroids, augment_obj_radius)
        if placer.placeable:  # Run placement search only when the current scene configuration is valid
            for obj_add_idx in np.random.permutation(match_obj_inst_labels):
                feasible_points, placeable = placer.compute_feasibility(match_obj_radius[obj_add_idx])
                if placeable:
                    place_loc = feasible_points[random.randint(0, feasible_points.shape[0] - 1)]
                    placer.update_placement(place_loc, match_obj_radius[obj_add_idx])

                    for k in match_add_triplet.keys():
                        if k not in ['pair_pos_idx_map', 'pair_pos_trans', 'pair_pos_fp_points', 'pair_pos_fp_lines', 'pair_pos_orig_scene_path', 'pair_pos_global_rot']:  # Centroids and translations are modified after placement
                            match_add_triplet[k].append(match_scene_triplet[k][obj_add_idx])
                        if k == 'pair_pos_idx_map':
                            match_add_triplet[k][keep_count + pose_perturb_count + curr_obj_add_count] = -1
                        if k == 'pair_pos_trans':
                            match_obj_scene_height = match_scene_triplet['pair_pos_trans'][obj_add_idx, 1]
                            place_trans = np.array([place_loc[0], match_obj_scene_height, place_loc[1]])
                            match_add_triplet[k].append(place_trans.reshape(-1))
                    curr_obj_add_count += 1

                if curr_obj_add_count == match_obj_add_count:
                    break

        # Update augmented scene with match scene objects
        if len(match_add_triplet['pair_pos_idx_map']) != 0:  # Non-empty object adding
            for k in augment_scene_triplet.keys():
                if k not in ['pair_pos_idx_map', 'pair_pos_fp_points', 'pair_pos_fp_lines', 'pair_pos_orig_scene_path', 'pair_pos_global_rot']:
                    match_add_triplet[k] = np.stack(match_add_triplet[k])
                    augment_scene_triplet[k] = np.concatenate([augment_scene_triplet[k], match_add_triplet[k]], axis=0)
                if k == 'pair_pos_idx_map':
                    for add_idx in match_add_triplet[k].keys():
                        augment_scene_triplet[k][add_idx] = -1

        # Save augmented scene
        save_scene_root = os.path.dirname(save_scene_files[scene_idx])
        if not os.path.exists(save_scene_root):
            os.makedirs(save_scene_root, exist_ok=True)

        # Check validity before saving
        is_valid = True
        if np.isnan(augment_scene_triplet['pair_pos_trans']).sum() != 0:
            is_valid = False
            break
        if np.isnan(augment_scene_triplet['pair_pos_rot']).sum() != 0:
            is_valid = False
            break
        if np.isnan(augment_scene_triplet['pair_pos_obj_scene_scales']).sum() != 0:
            is_valid = False
            break

        if is_valid:
            np.savez_compressed(save_scene_files[scene_idx], **augment_scene_triplet)

            # Optionally visualize scenes
            if args.visualize_scene:
                orig_dense = build_dense_3d_scene(scene_path, 'pair_pos')
                augment_dense = build_dense_3d_scene(save_scene_files[scene_idx], 'pair_pos')
                augment_dense = o3d_geometry_list_shift(augment_dense, shift_amount=[10., 0., 0.])

                o3d.visualization.draw_geometries(orig_dense + augment_dense)
            if args.visualize_gt_match:
                orig_dense = build_dense_3d_scene(scene_path, 'pair_pos')
                augment_dense = build_dense_3d_scene(save_scene_files[scene_idx], 'pair_pos')
                augment_dense = o3d_geometry_list_shift(augment_dense, shift_amount=[10., 0., 0.])

                # Sample surface points of ground-truth match objects
                num_obj_query = 200
                orig_query_pcd = generate_obj_surface_points([scene_path], 'pair_pos', num_obj_query, args.point_sample_root)
                augment_query_pcd = generate_obj_surface_points([save_scene_files[scene_idx]], 'pair_pos', num_obj_query, args.point_sample_root)

                full_orig_query_points = orig_query_pcd.points_packed().cpu().numpy()
                full_augment_query_points = augment_query_pcd.points_packed().cpu().numpy()
                full_orig_query_inst = orig_query_pcd.features_packed().cpu().numpy()[:, 0]
                full_augment_query_inst = augment_query_pcd.features_packed().cpu().numpy()[:, 0]

                vis_orig_query_points = []
                vis_augment_query_points = []
                for augment_idx, orig_idx in augment_scene_triplet['pair_pos_idx_map'].items():
                    if orig_idx != -1:
                        vis_orig_query_points.append(full_orig_query_points[full_orig_query_inst == orig_idx])
                        vis_augment_query_points.append(full_augment_query_points[full_augment_query_inst == augment_idx])

                vis_orig_query_points = np.concatenate(vis_orig_query_points, axis=0)
                vis_augment_query_points = np.concatenate(vis_augment_query_points, axis=0)

                idx_color = np.linspace(0., 1., vis_orig_query_points.shape[0])  # (N_query, )
                idx_color = colormaps['jet'](idx_color, alpha=False, bytes=False)[:, :3]

                # Sort points for visualization
                x_sort_idx = np.argsort(vis_augment_query_points[:, 0])
                y_sort_idx = np.argsort(vis_augment_query_points[x_sort_idx, 1])
                z_sort_idx = np.argsort(vis_augment_query_points[x_sort_idx[y_sort_idx], 2])
                vis_augment_query_points = vis_augment_query_points[x_sort_idx[y_sort_idx[z_sort_idx]]]
                vis_orig_query_points = vis_orig_query_points[x_sort_idx[y_sort_idx[z_sort_idx]]]

                vis_augment_query_pcd = o3d.geometry.PointCloud()
                vis_orig_query_pcd = o3d.geometry.PointCloud()

                vis_augment_query_pcd.points = o3d.utility.Vector3dVector(vis_augment_query_points)
                vis_orig_query_pcd.points = o3d.utility.Vector3dVector(vis_orig_query_points)
                vis_augment_query_pcd.colors = o3d.utility.Vector3dVector(idx_color)
                vis_orig_query_pcd.colors = o3d.utility.Vector3dVector(idx_color)

                vis_augment_query = [vis_augment_query_pcd]
                vis_orig_query = [vis_orig_query_pcd]
                vis_augment_query = o3d_geometry_list_shift(vis_augment_query, shift_amount=[10., 0., 0.])
                o3d.visualization.draw_geometries(orig_dense + augment_dense + vis_augment_query + vis_orig_query)
        else:
            subprocess.run(['rm', '-rf', save_scene_root])
