import argparse
import torch
import numpy as np
import random
import os
from glob import glob
from tqdm import tqdm
import json
from trimesh.sample import sample_surface
from threed_front.utils import trimesh_load_with_postprocess, farthest_point_down_sample
from threed_front.invalid_files import PROBLEMATIC_OBJ_ID
from PIL import Image
import open3d as o3d


if __name__ == '__main__':
    """
        Extract and save point features along with point coordinates for feature extraction. Results will be saved for each point cloud as an (N_shapes, 3 + D_emb) array where D_emb is the dimension of the embedding
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_root", help="Root directory containing 3D-FUTURE meshes", default="./data/3D-FUTURE-model/")
    parser.add_argument("--obj_json", help=".json file containing information on 3D-FUTURE meshes", default="./data/3D-FUTURE-model/model_info.json")
    parser.add_argument("--save_root", help="Root directory to save sampled points", default="./data/3d_future_point_samples/")
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

    # Post-process 3D-FUTURE object directory
    with open(args.obj_json, 'r') as f:
        obj_list = json.load(f)

    obj_dirs = [obj for obj in sorted(glob(os.path.join(args.obj_root, "*"))) if ('.json' not in obj) and ('.py' not in obj)]
    num_objs = len(obj_dirs)
    pbar = tqdm(total=num_objs, desc="Feature Extraction")
    for obj_idx, obj_dir in enumerate(obj_dirs):
        obj_id = obj_dir.split("/")[-1]

        # Skip invalid objects
        if obj_id in PROBLEMATIC_OBJ_ID:
            continue

        obj_mesh_path = os.path.join(obj_dir, "raw_model.obj")
        obj_mesh = trimesh_load_with_postprocess(obj_mesh_path, 'bottom_crop')

        point_sample_list = []
        for num_sample in args.save_num_samples:
            point_samples = sample_surface(obj_mesh, args.mesh_num_samples)[0]
            point_samples = farthest_point_down_sample(point_samples, num_sample, max_input_size=args.mesh_num_samples)
            point_sample_list.append(point_samples)
            sample_save_root = os.path.join(args.save_root, f"sample_{num_sample}")
            sample_save_path = os.path.join(sample_save_root, obj_id + ".npy")
            np.save(sample_save_path, point_samples)

        if args.visualize_obj:
            texture_path = obj_mesh_path.replace("raw_model.obj", "texture.png")
            if not os.path.exists(texture_path):
                texture_path = obj_mesh_path.replace("raw_model.obj", "texture.jpg")
            obj_mesh.visual.material.image = Image.open(texture_path)

            # Prepare object mesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
            uvs = obj_mesh.visual.uv
            uvs[:, 1] = 1 - uvs[:, 1]
            triangles_uvs = []
            for i in range(3):
                triangles_uvs.append(uvs[obj_mesh.faces[:, i]].reshape(-1, 1, 2))
            triangles_uvs = np.concatenate(triangles_uvs, axis=1).reshape(-1, 2)

            o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
            texture_image = np.asarray(obj_mesh.visual.material.image)
            if texture_image.shape[-1] == 2:  # Greyscale textures should be converted to RGBA
                texture_image = np.stack([texture_image[..., 0]] * 3 + [texture_image[..., 1]], axis=-1)
            o3d_mesh.textures = [o3d.geometry.Image(texture_image)]
            o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(obj_mesh.faces))

            o3d_pcd_list = []
            for p_idx, point_samples in enumerate(point_sample_list):
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(point_samples)
                o3d_pcd.translate(np.array([(p_idx + 1) * args.vis_margin, 0., 0.], dtype=np.float64).T)
                o3d_pcd_list.append(o3d_pcd)
            o3d.visualization.draw_geometries([o3d_mesh] + o3d_pcd_list)

        pbar.update(1)
