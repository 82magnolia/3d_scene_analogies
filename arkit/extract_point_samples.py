import argparse
import torch
import numpy as np
import random
import os
from glob import glob
from tqdm import tqdm
import trimesh
from trimesh.sample import sample_surface
from arkit.utils import farthest_point_down_sample
import open3d as o3d


if __name__ == '__main__':
    """
        Extract and save point features along with point coordinates for feature extraction. Results will be saved for each point cloud as an (N_shapes, 3 + D_emb) array where D_emb is the dimension of the embedding
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_root", help="Root directory containing ARKIT object meshes", default="./data/arkit_scenes_processed/objects/")
    parser.add_argument("--save_root", help="Root directory to save sampled points", default="./data/arkit_scenes_point_samples/")
    parser.add_argument("--mesh_num_samples", help="Number of points to initially subsample from mesh", default=10000, type=int)
    parser.add_argument("--save_num_samples", help="List containing number of points to subsample from initial point cloud", default=[50], type=int, nargs="+")
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)
    parser.add_argument("--visualize_obj", help="Optionally visualize point sampling", action="store_true")
    parser.add_argument("--vis_margin", help="Amount of margins to apply for objects during visualization", default=1., type=float)
    args = parser.parse_args()

    # Fix seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)

    for num_sample in args.save_num_samples:
        sample_save_root = os.path.join(args.save_root, f"sample_{num_sample}")
        if not os.path.exists(sample_save_root):
            os.makedirs(sample_save_root, exist_ok=True)

    train_obj_dirs = sorted(glob(os.path.join(args.obj_root, "train", "*.ply")))
    test_obj_dirs = sorted(glob(os.path.join(args.obj_root, "test", "*.ply")))
    obj_dirs = train_obj_dirs + test_obj_dirs
    num_objs = len(obj_dirs)
    pbar = tqdm(total=num_objs, desc="Feature Extraction")
    for obj_idx, obj_dir in enumerate(obj_dirs):
        obj_id = obj_dir.split("/")[-1].replace(".ply", "")
        obj_mesh = trimesh.load_mesh(obj_dir)

        point_sample_list = []
        for num_sample in args.save_num_samples:
            point_samples = sample_surface(obj_mesh, args.mesh_num_samples)[0]
            point_samples = farthest_point_down_sample(point_samples, num_sample, args.mesh_num_samples)
            point_sample_list.append(point_samples)
            sample_save_root = os.path.join(args.save_root, f"sample_{num_sample}")
            sample_save_path = os.path.join(sample_save_root, obj_id + ".npy")
            np.save(sample_save_path, point_samples)

        if args.visualize_obj:
            o3d_mesh = o3d.io.read_triangle_mesh(obj_dir)
            o3d_pcd_list = []
            for p_idx, point_samples in enumerate(point_sample_list):
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(point_samples)
                o3d_pcd.translate(np.array([(p_idx + 1) * args.vis_margin, 0., 0.], dtype=np.float64).T)
                o3d_pcd_list.append(o3d_pcd)
            o3d.visualization.draw_geometries([o3d_mesh] + o3d_pcd_list)

        pbar.update(1)
