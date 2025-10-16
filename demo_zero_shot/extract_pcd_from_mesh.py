import open3d as o3d
from glob import glob
import argparse
import os
from threed_front.data_utils import (
    trimesh_load_with_postprocess,
    sample_surface
)
import numpy as np


if __name__ == "__main__":
    """
        To run 3D scene analogy estimation on a pair of point clouds, we need two point clouds to exist for the same scene:

            i) Colored point cloud: Saved in a .ply format containing color [x y z r g b].
            ii) Instance point cloud: Saved in a .ply format containing instance information [x y z instance_id instance_id instance_id].

        We assume each mesh to be saved in the following structure:
            ./mesh_folder/obj_1/model.ply  # Mesh for object 1
            ./mesh_folder/obj_2/model.ply  # Mesh for object 2
            ...
    """
    parser = argparse.ArgumentParser()
    # General configs
    parser.add_argument("--scene_dir", help="Path to directory containing mesh folders", type=str)
    parser.add_argument("--save_pcd_path", help="Path to saving extracted point cloud", required=True)
    parser.add_argument("--inst_num_pts", help="Number of points per instance", type=int, default=10000)
    parser.add_argument("--global_scale", help="Amount of global scale to apply", type=float, default=3.)
    parser.add_argument("--global_rot", help="Type of global rotation matrix to apply", type=str, default="gso")

    args = parser.parse_args()

    if args.global_rot == "gso":  # Global rotation matrix used for converting Google Scanned Objects (GSO) to 3D-FRONT
        global_rot = np.array([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., -1., 0.]
        ])
    else:
        raise NotImplementedError("Other global rotations not supported") 

    mesh_folder_list = glob(os.path.join(args.scene_dir, "obj_*"))
    mesh_path_list = [os.path.join(mf, "model.ply") for mf in mesh_folder_list]

    points_list = []
    colors_list = []
    for mesh_path in mesh_path_list:
        tr_mesh = trimesh_load_with_postprocess(mesh_path)
        points, _, colors = sample_surface(tr_mesh, args.inst_num_pts, sample_color=True)
        points_rot = points @ global_rot.T
        points_scaled = points_rot * args.global_scale

        points_list.append(points_scaled)
        colors_list.append(colors[:, :-1] / 255.)

    scene_points = np.concatenate(points_list, axis=0)
    scene_colors = np.concatenate(colors_list, axis=0)

    # Zero-center scene points
    scene_points[:, 1] = scene_points[:, 1] - scene_points[:, 1].min()

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)

    o3d.io.write_point_cloud(args.save_pcd_path, scene_pcd)
