import argparse
from glob import glob
from tqdm import tqdm
import os
import open3d as o3d
import numpy as np
import json
import hashlib
from scipy.spatial import ConvexHull
from point_features.utils import str2cls_arkit


ARKIT_TO_TDFRONT = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def compute_box_3d(scale, transform, rotation):
    scales = [i / 2 for i in scale]
    l, h, w = scales
    center = np.reshape(transform, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    raw_corners_3d = np.vstack([x_corners, y_corners, z_corners]).T
    corners_3d = raw_corners_3d @ rotation.T
    corners_3d = corners_3d + transform
    return corners_3d


if __name__ == '__main__':
    """
    NOTE: This script parses mesh files in ARKiT and outputs individual 3D object files located at canonical poses, along with scene files containing layout information.
    In a nutshell, the script converts ARKiT to a file format akin to 3D-FRONT.
    Generated assets are saved under args.save_root in the following format:

    scenes/
        train/
            SCENE_ID.npz  # .npz file containing scene information (object pose, floorplan, etc.)

        test/
            SCENE_ID.npz  # .npz file containing scene information (object pose, floorplan, etc.)

    objects/
        train/
            OBJ_ID.ply  # .ply file containing object geometry (class label is saved in object name)

        test/
            OBJ_ID.ply  # .ply file containing object geometry (class label is saved in object name)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", help="Root directory containing scenes to load", default="./data/arkit_scenes/")
    parser.add_argument("--save_root", help="Root directory to save object and scene information", default="./data/arkit_scenes_processed/")
    parser.add_argument("--visualize_save_scene", help="Optionally visualize objects and floorplans being saved", action="store_true")
    parser.add_argument("--visualize_hull", help="Optionally visualize scene mesh and floorplans being saved", action="store_true")
    args = parser.parse_args()

    # List up directories used for file saving
    save_scene_root = os.path.join(args.save_root, 'scenes')
    save_obj_root = os.path.join(args.save_root, 'objects')

    train_save_scene_root = os.path.join(save_scene_root, 'train')
    test_save_scene_root = os.path.join(save_scene_root, 'test')

    train_save_obj_root = os.path.join(save_obj_root, 'train')
    test_save_obj_root = os.path.join(save_obj_root, 'test')

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)

        # Make file structure
        os.makedirs(save_scene_root, exist_ok=True)
        os.makedirs(save_obj_root, exist_ok=True)
        os.makedirs(train_save_scene_root, exist_ok=True)
        os.makedirs(test_save_scene_root, exist_ok=True)
        os.makedirs(train_save_obj_root, exist_ok=True)
        os.makedirs(test_save_obj_root, exist_ok=True)

    scene_folders = sorted(glob(os.path.join(args.scene_root, "**/*")))
    scene_ids = [os.path.basename(path.rstrip('/')) for path in scene_folders]
    scene_mesh_files = [os.path.join(path, scene_id + "_3dod_mesh.ply") for path, scene_id in zip(scene_folders, scene_ids)]
    scene_annot_files = [os.path.join(path, scene_id + "_3dod_annotation.json") for path, scene_id in zip(scene_folders, scene_ids)]

    train_scene_mesh_files = [path for path in scene_mesh_files if 'train' in path]
    train_scene_annot_files = [path for path in scene_annot_files if 'train' in path]

    test_scene_mesh_files = [path for path in scene_mesh_files if 'test' in path]
    test_scene_annot_files = [path for path in scene_annot_files if 'test' in path]

    num_scenes = len(scene_mesh_files)
    print(f"Number of scenes: {num_scenes}")
    total_num_obj = 0
    hasher = hashlib.md5()
    PROBLEMATIC_SCENE_ID = []
    for mesh_path, annot_path in tqdm(zip(scene_mesh_files, scene_annot_files), total=num_scenes, desc="Asset Extraction"):
        scene_id = os.path.dirname(mesh_path).split('/')[-1]
        if 'train' in mesh_path:
            scene_type = 'train'
        else:
            scene_type = 'test'

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ ARKIT_TO_TDFRONT.T)

        # Add reversed trianges for clear visualization
        mesh.triangles = o3d.utility.Vector3iVector(np.concatenate([np.asarray(mesh.triangles), np.asarray(mesh.triangles)[:, [0, 2, 1]]], axis=0))

        with open(annot_path, 'rb') as f:
            annot = json.load(f)

        scene_dict = {'trans': [], 'rot': [], 'obj_ids': [], 'obj_classes': [], 'obj_classes_str': [], 'fp_points': None, 'fp_lines': None}  # Dictionary for saving scene information

        crop_mesh_list = []
        box3d_list = []
        for annot_data in annot['data']:
            # Save objects
            # NOTE: ARKiT Scenes annotations use X' = X @ R + T transformation for (N, 3) matrix X
            rotation = np.array(annot_data["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3).T  # Transposed to support X' = X @ R.T + T formulation
            transform = np.array(annot_data["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
            bbox_size = np.array(annot_data["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)  # NOTE: This is object size in canonical frame

            # NOTE: We only need to transform rotations and translations to TDFRONT coordinate frames (and keep scale)
            rotation = ARKIT_TO_TDFRONT @ rotation
            transform = transform @ ARKIT_TO_TDFRONT.T
            box3d = compute_box_3d(bbox_size.reshape(3).tolist(), transform, rotation)
            box3d_list.append(box3d)

            oriented_bbox = o3d.geometry.OrientedBoundingBox()
            oriented_bbox = oriented_bbox.create_from_points(o3d.utility.Vector3dVector(box3d))
            crop_mesh = mesh.crop(oriented_bbox)
            canonical_crop_mesh = o3d.geometry.TriangleMesh(crop_mesh)

            # Move object back to canonical pose
            transformed_vertices = np.asarray(crop_mesh.vertices)
            canonical_vertices = (transformed_vertices - transform) @ rotation
            canonical_crop_mesh.vertices = o3d.utility.Vector3dVector(canonical_vertices)

            hasher.update(str(total_num_obj).encode("utf-8"))
            obj_label = annot_data["label"]
            obj_id = obj_label + "_" + hasher.hexdigest()

            # Check if object label is valid, and abort saving if it is invalid
            if str2cls_arkit(obj_label, return_type='int') == -1:
                PROBLEMATIC_SCENE_ID.append(scene_id)
                continue

            # Skip object if no labels exist
            if len(crop_mesh.vertices) == 0:
                PROBLEMATIC_SCENE_ID.append(scene_id)
                continue

            obj_save_path = os.path.join(save_obj_root, scene_type, obj_id + ".ply")
            o3d.io.write_triangle_mesh(obj_save_path, canonical_crop_mesh)
            crop_mesh_list.append(crop_mesh)

            # Save scene
            scene_dict['trans'].append(transform.reshape(-1))
            scene_dict['rot'].append(rotation)
            scene_dict['obj_ids'].append(obj_id)
            scene_dict['obj_classes_str'].append(obj_label)
            scene_dict['obj_classes'].append(str2cls_arkit(obj_label, return_type='int'))

            total_num_obj += 1

        # Skip scene saving if no objects exist
        if len(box3d_list) == 0:
            PROBLEMATIC_SCENE_ID.append(scene_id)
            continue

        full_box3d = np.concatenate(box3d_list, axis=0)
        scene_dict['trans'] = np.stack(scene_dict['trans'], axis=0)
        scene_dict['rot'] = np.stack(scene_dict['rot'], axis=0)
        scene_dict['obj_ids'] = np.array(scene_dict['obj_ids'])
        scene_dict['obj_classes_str'] = np.array(scene_dict['obj_classes_str'])
        scene_dict['obj_classes'] = np.array(scene_dict['obj_classes'])

        # Generate scene floorplans by computing 2D convex hull
        mesh_2d_coords = np.asarray(mesh.vertices)[:, [0, 2]]
        cvxhull_2d = ConvexHull(mesh_2d_coords)
        cvxhull_coords = mesh_2d_coords[cvxhull_2d.vertices]

        # Choose ground and ceil level from box3d information instead of scans as the geometry is noisy
        ground_level = full_box3d[:, 1].min()
        ceil_level = full_box3d[:, 1].max()

        ground_fp_points = np.zeros([cvxhull_coords.shape[0], 3])
        ground_fp_points[:, [0, 2]] = cvxhull_coords
        ground_fp_points[:, 1] = ground_level
        ceil_fp_points = np.zeros([cvxhull_coords.shape[0], 3])
        ceil_fp_points[:, [0, 2]] = cvxhull_coords
        ceil_fp_points[:, 1] = ceil_level

        ground_fp_lines = np.stack([np.arange(ground_fp_points.shape[0]), np.roll(np.arange(ground_fp_points.shape[0]), shift=-1)], axis=1)
        ceil_fp_lines = np.stack([np.arange(ceil_fp_points.shape[0]) + ceil_fp_points.shape[0], np.roll(np.arange(ceil_fp_points.shape[0]) + ceil_fp_points.shape[0], shift=-1)], axis=1)
        inter_fp_lines = np.concatenate([ground_fp_lines[:, 0:1], ceil_fp_lines[:, 0:1]], axis=-1)

        fp_points = np.concatenate([ground_fp_points, ceil_fp_points], axis=0)
        fp_lines = np.concatenate([ground_fp_lines, ceil_fp_lines, inter_fp_lines], axis=0)

        scene_dict['fp_points'] = fp_points
        scene_dict['fp_lines'] = fp_lines

        fp_lineset = o3d.geometry.LineSet()
        fp_lineset.points = o3d.utility.Vector3dVector(fp_points)
        fp_lineset.lines = o3d.utility.Vector2iVector(fp_lines)
        if args.visualize_save_scene:
            o3d.visualization.draw_geometries(crop_mesh_list + [fp_lineset])
        if args.visualize_hull:
            o3d.visualization.draw_geometries([mesh] + [fp_lineset])

        scene_save_path = os.path.join(save_scene_root, scene_type, scene_id + ".npz")
        np.savez_compressed(scene_save_path, **scene_dict)

    print("Finished asset creation! Problematic scenes: ")
    for prob_scene_id in PROBLEMATIC_SCENE_ID:
        print(prob_scene_id)
