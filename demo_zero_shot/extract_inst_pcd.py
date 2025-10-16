import numpy as np
import open3d as o3d
import argparse
import os


if __name__ == "__main__":
    """
        To run 3D scene analogy estimation on a pair of point clouds, we need two point clouds to exist for the same scene:

            i) Colored point cloud: Saved in a .ply format containing color [x y z r g b].
            ii) Instance point cloud: Saved in a .ply format containing instance information [x y z instance_id instance_id instance_id].

        We assume each point cloud to be saved in the following structure:
            ./pcd_folder/pcd.ply  # Colored point cloud
            ./pcd_folder/inst_pcd.ply  # Instance labels
            ...
    """
    parser = argparse.ArgumentParser()
    # General configs
    parser.add_argument("--pcd_path", help="Path to point cloud for obtaining instance labels", type=str)
    parser.add_argument("--inst_num_pts", help="Number of points per instance", type=int, default=10000)

    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    num_obj_inst = round(points.shape[0] / args.inst_num_pts)

    id_list = []
    for obj_id in range(num_obj_inst):
        obj_points = points[obj_id * args.inst_num_pts: (obj_id + 1) * args.inst_num_pts]
        id_list.append(np.ones_like(np.asarray(obj_points)) * obj_id / num_obj_inst)
    ids = np.concatenate(id_list, axis=0)

    inst_pcd = o3d.geometry.PointCloud()
    inst_pcd.points = o3d.utility.Vector3dVector(points)
    inst_pcd.colors = o3d.utility.Vector3dVector(ids)

    log_dir = os.path.dirname(args.pcd_path)
    o3d.io.write_point_cloud(os.path.join(log_dir, "inst_scene_pcd.ply"), inst_pcd)
