import argparse
import torch
import numpy as np
import random
import os
import open3d as o3d
from threed_front.global_field_match import AffineMatcher
from threed_front.local_field_match import PointMatcher
from threed_front.utils import (
    farthest_point_down_sample,
)
from demo_zero_shot.vis_utils import (
    visualize_input,
    visualize_local_transform
)
from pytorch3d.structures import Pointclouds


if __name__ == '__main__':
    """
        Run 3D scene analogy estimation on a pair of point clouds. Assume two point clouds exist for the same scene:

            i) Colored point cloud: Saved in a .ply format containing color [x y z r g b].
            ii) Instance point cloud: Saved in a .ply format containing instance information [x y z instance_id instance_id instance_id].

        We assume each point cloud to be saved in the following structure:
            ./pcd_folder/pcd.ply  # Colored point cloud
            ./pcd_folder/inst_pcd.ply  # Instance labels
            ...
    """
    parser = argparse.ArgumentParser()
    # General configs
    parser.add_argument("--log_dir", help="Log directory for saving experiment results", default="./log/")
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)

    # Dataset configs
    parser.add_argument("--pcd_tgt", help="Path to colored point cloud of target scene", required=True, type=str)
    parser.add_argument("--pcd_ref", help="Path to colored point cloud of reference scene", required=True, type=str)
    parser.add_argument("--roi_ids", help="Instance IDs in target scene to use for scene analogy estimation", default=None, nargs="+", type=int)
    parser.add_argument("--scene_sample_points", type=int, help="Number of points to sample per object model for scene generation", default=50)
    parser.add_argument("--num_query", help="Number of query point locations to exploit per scene for local feature extraction", default=50, type=int)
    parser.add_argument("--field_type", help="Field type to use for finding 3D scene analogies", default="vn", type=str)

    # Feature field configs
    parser.add_argument("--load_local_feature_field", help="Load pre-trained local feature field", default=None, type=str)
    parser.add_argument("--load_global_feature_field", help="Load pre-trained global feature field", default=None, type=str)
    parser.add_argument("--max_infer_point", help="Maximum number of query points to process per inference", default=1000, type=int)

    # Training configs
    parser.add_argument("--num_vert_split", help="Number of vertical splits to make for floorplan queries", default=1, type=int)
    parser.add_argument("--force_up_margin", help="Forced upper margin for floorplan queries (defaults to using maximum height of walls)", default=0.5, type=float)
    parser.add_argument("--force_low_margin", help="Forced lower margin for floorplan queries (defaults to using minimum height of walls)", default=0.5, type=float)
    parser.add_argument("--num_classes", help="Number of semantics classes in objects", default=8, type=int)
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
    parser.add_argument("--local_margin_thres", help="Margin value to keep local transforms from the minimum cost transform", type=float, default=0.05)
    parser.add_argument("--local_threshold_method", help="Local thresholding method to use", type=str, default="valid")
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
    parser.add_argument("--save_query_mesh", help="Optionally save query points as meshes when save_transfer is on", action="store_true")

    # Visualization configs
    parser.add_argument("--vis_input_mode", help="Mode for visualizing input scene pairs", default=None, type=str)
    parser.add_argument("--vis_global_match_mode", help="Mode for visualizing global matches during evaluation", default=None, type=str)
    parser.add_argument("--vis_local_match_mode", help="Mode for visualizing local matches during evaluation", default=None, type=str)
    parser.add_argument("--vis_margin", help="Amount of margins to apply for reference scene during visualization", default=5., type=float)
    parser.add_argument("--vis_method", help="Type of visualization to perform", default="screen", type=str)
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

    # Initiate feature field
    if args.field_type in ["distance"]:
        # Setup local feature field
        assert args.load_local_feature_field is not None
        print(f"Loading model from {args.load_local_feature_field}")
        local_feature_field = torch.load(args.load_local_feature_field).to(device)

        # Freeze parameters for local feature field
        local_feature_field = local_feature_field.eval()
        for param in local_feature_field.parameters():
            param.requires_grad = False
        local_feature_field.to(device)

        # Setup global feature field
        assert args.load_global_feature_field is not None
        print(f"Loading model from {args.load_global_feature_field}")
        global_feature_field = torch.load(args.load_global_feature_field).to(device)

        # Freeze parameters for local feature field
        global_feature_field = global_feature_field.eval()
        for param in global_feature_field.parameters():
            param.requires_grad = False
        global_feature_field.to(device)
    else:
        raise NotImplementedError("Other fields currently unsupported")

    # Load point cloud
    pcd_tgt = o3d.io.read_point_cloud(args.pcd_tgt)
    pcd_ref = o3d.io.read_point_cloud(args.pcd_ref)
    pcd_tgt_points = np.asarray(pcd_tgt.points)
    pcd_ref_points = np.asarray(pcd_ref.points)
    pcd_tgt_colors = np.asarray(pcd_tgt.colors)
    pcd_ref_colors = np.asarray(pcd_ref.colors)

    # Load instance point cloud
    pcd_tgt_name = os.path.basename(args.pcd_tgt).replace(".ply", "")
    pcd_ref_name = os.path.basename(args.pcd_ref).replace(".ply", "")

    pcd_tgt_inst_path = args.pcd_tgt.replace(pcd_tgt_name, "inst_" + pcd_tgt_name)
    pcd_ref_inst_path = args.pcd_ref.replace(pcd_ref_name, "inst_" + pcd_ref_name)
    pcd_tgt_inst = o3d.io.read_point_cloud(pcd_tgt_inst_path)
    pcd_ref_inst = o3d.io.read_point_cloud(pcd_ref_inst_path)
    pcd_tgt_inst = np.asarray(pcd_tgt_inst.colors)[:, 0]
    pcd_ref_inst = np.asarray(pcd_ref_inst.colors)[:, 0]
    pcd_tgt_inst_labels = np.unique(pcd_tgt_inst)
    pcd_ref_inst_labels = np.unique(pcd_ref_inst)

    # Un-normalize instance IDs
    num_tgt_inst = pcd_tgt_inst_labels.shape[0]
    num_ref_inst = pcd_ref_inst_labels.shape[0]
    pcd_tgt_inst_labels = np.round(pcd_tgt_inst_labels * num_tgt_inst).astype(int)
    pcd_ref_inst_labels = np.round(pcd_ref_inst_labels * num_ref_inst).astype(int)
    pcd_tgt_inst = np.round(pcd_tgt_inst * num_tgt_inst).astype(int)
    pcd_ref_inst = np.round(pcd_ref_inst * num_ref_inst).astype(int)

    # Load per-instance point cloud features
    pcd_tgt_inst_feat_path = args.pcd_tgt.replace(pcd_tgt_name + ".ply", f"{args.field_type}_" + pcd_tgt_name) + ".npz"
    pcd_ref_inst_feat_path = args.pcd_ref.replace(pcd_ref_name + ".ply", f"{args.field_type}_" + pcd_ref_name) + ".npz"

    semantics_emb_tgt = {}
    semantics_emb_ref = {}

    # Build 3D scene (split point cloud from instance IDs)
    scene_tgt = {f"obj_{obj_id}": {} for obj_id in pcd_tgt_inst_labels}
    scene_ref = {f"obj_{obj_id}": {} for obj_id in pcd_ref_inst_labels}
    scene_obj_id_dict = {"pair_pos": [[]], "pos": [[]]}  # NOTE: pair_pos corresponds to tgt and pos corresponds to ref (this is used to respect the original convention in 3D Scene Analogies)

    for obj_id in pcd_tgt_inst_labels:
        obj_points = pcd_tgt_points[pcd_tgt_inst == obj_id]
        obj_colors = pcd_tgt_colors[pcd_tgt_inst == obj_id]
        scene_tgt[f"obj_{obj_id}"]["points"] = obj_points
        scene_tgt[f"obj_{obj_id}"]["colors"] = obj_colors
        scene_tgt[f"obj_{obj_id}"]["obj_id"] = "tgt_" + f"obj_{obj_id}"
        scene_obj_id_dict["pair_pos"][0].append("tgt_" + f"obj_{obj_id}")

        # Sample scene points (for field estimation) and query points
        scene_tgt[f"obj_{obj_id}"]["scene_points"] = farthest_point_down_sample(obj_points, args.scene_sample_points)

        if obj_id in args.roi_ids:
            scene_tgt[f"obj_{obj_id}"]["query_points"], sample_idx = farthest_point_down_sample(obj_points, args.num_query, return_idx=True)
            scene_tgt[f"obj_{obj_id}"]["query_colors"] = obj_colors[sample_idx]

    for obj_id in pcd_ref_inst_labels:
        obj_points = pcd_ref_points[pcd_ref_inst == obj_id]
        obj_colors = pcd_ref_colors[pcd_ref_inst == obj_id]
        scene_ref[f"obj_{obj_id}"]["points"] = obj_points
        scene_ref[f"obj_{obj_id}"]["colors"] = obj_colors
        scene_ref[f"obj_{obj_id}"]["obj_id"] = "ref_" + f"obj_{obj_id}"
        scene_obj_id_dict["pos"][0].append("ref_" + f"obj_{obj_id}")

        # Sample scene points (for field estimation) and query points
        scene_ref[f"obj_{obj_id}"]["scene_points"] = farthest_point_down_sample(obj_points, args.scene_sample_points)
        scene_ref[f"obj_{obj_id}"]["query_points"], sample_idx = farthest_point_down_sample(obj_points, args.num_query, return_idx=True)
        scene_ref[f"obj_{obj_id}"]["query_colors"] = obj_colors[sample_idx]

    # Instantiate global / local matcher
    if args.global_matcher_type == 'affine':
        global_matcher = AffineMatcher(args, device, global_feature_field)
    else:
        raise NotImplementedError("Other global matchers not supported")

    if args.local_matcher_type == 'point':
        local_matcher = PointMatcher(args, device, local_feature_field)
    else:
        raise NotImplementedError("Other local matchers not supported")

    # Build scene point clouds
    tgt_dict = {'points': [], 'feats': []}
    tgt_query_dict = {'points': [], 'feats': []}
    ref_dict = {'points': [], 'feats': []}
    ref_query_dict = {'points': [], 'feats': []}

    for obj_id in pcd_tgt_inst_labels:
        points = torch.from_numpy(scene_tgt[f"obj_{obj_id}"]["scene_points"])
        tgt_dict['points'].append(points)

        inst_feats = torch.ones_like(points[:, 0:1]) * obj_id
        sem_feats = torch.ones_like(points[:, 0:1]) * 0  # NOTE: Semantic labels are hard-coded, assuming we don't know them

        # Extract point-level features (optional)
        if args.field_type in ["distance"]:
            point_feats = torch.zeros(points.shape[0], 0)
        else:
            raise NotImplementedError("Other fields currently unsupported")

        tgt_dict['feats'].append(torch.cat([inst_feats, sem_feats, point_feats], dim=-1))

        # Extract query point information
        if obj_id in args.roi_ids:
            query_points = torch.from_numpy(scene_tgt[f"obj_{obj_id}"]["query_points"])
            query_inst_feats = torch.ones_like(query_points[:, 0:1]) * obj_id
            query_colors = torch.from_numpy(scene_tgt[f"obj_{obj_id}"]["query_colors"])

            tgt_query_dict['points'].append(query_points)
            tgt_query_dict['feats'].append(torch.cat([query_inst_feats, query_colors], dim=-1))

    for obj_id in pcd_ref_inst_labels:
        points = torch.from_numpy(scene_ref[f"obj_{obj_id}"]["scene_points"])
        ref_dict['points'].append(points)

        inst_feats = torch.ones_like(points[:, 0:1]) * obj_id
        sem_feats = torch.ones_like(points[:, 0:1]) * 0  # NOTE: Semantic labels are hard-coded, assuming we don't know them

        # Extract point-level features (optional)
        if args.field_type in ["distance"]:
            point_feats = torch.zeros(points.shape[0], 0)
        else:
            raise NotImplementedError("Other fields currently unsupported")

        ref_dict['feats'].append(torch.cat([inst_feats, sem_feats, point_feats], dim=-1))

        query_points = torch.from_numpy(scene_ref[f"obj_{obj_id}"]["query_points"])
        query_inst_feats = torch.ones_like(query_points[:, 0:1]) * obj_id
        query_colors = torch.from_numpy(scene_ref[f"obj_{obj_id}"]["query_colors"])

        ref_query_dict['points'].append(query_points)
        ref_query_dict['feats'].append(torch.cat([query_inst_feats, query_colors], dim=-1))

    tgt_points = torch.cat(tgt_dict['points'], dim=0).float()
    tgt_feats = torch.cat(tgt_dict['feats'], dim=0).float()
    ref_points = torch.cat(ref_dict['points'], dim=0).float()
    ref_feats = torch.cat(ref_dict['feats'], dim=0).float()
    scene_tgt_pcd = Pointclouds(points=[tgt_points], features=[tgt_feats]).to(device)
    scene_ref_pcd = Pointclouds(points=[ref_points], features=[ref_feats]).to(device)

    tgt_query_points = torch.cat(tgt_query_dict['points'], dim=0).float()
    tgt_query_feats = torch.cat(tgt_query_dict['feats'], dim=0).float()
    ref_query_points = torch.cat(ref_query_dict['points'], dim=0).float()
    ref_query_feats = torch.cat(ref_query_dict['feats'], dim=0).float()
    query_tgt = Pointclouds(points=[tgt_query_points], features=[tgt_query_feats]).to(device)
    query_ref = Pointclouds(points=[ref_query_points], features=[ref_query_feats]).to(device)

    num_batch_scenes = 1  # NOTE: Currently the code only supports a single scene pair

    # Global alignment: list up instances to perform global matching
    if getattr(global_feature_field.cfg, "semantics_emb_type", "class_vector") == 'class_vector':
        global_transform_list, global_inst_match_list = global_matcher.find_transforms(query_tgt, query_ref, scene_tgt_pcd.to(device), scene_ref_pcd.to(device))
    else:
        raise NotImplementedError("Other embeddings not supported")

    # Estimate local transform
    if args.skip_local_matching:
        assert args.vis_local_match_mode is None
        local_transform_list = [[] for _ in range(num_batch_scenes)]
    else:
        if getattr(local_feature_field.cfg, "semantics_emb_type", "class_vector") == 'class_vector':
            local_transform_list, local_inst_match_list = local_matcher.find_transforms(global_transform_list, global_inst_match_list, query_tgt, query_ref, scene_tgt_pcd.to(device), scene_ref_pcd.to(device))
        else:
            raise NotImplementedError("Other embeddings not supported")

    # Cache estimated transforms
    estim_transform_list = [
        transform.cpu() for transform in local_transform_list[0]
    ]

    for t_idx, _ in enumerate(estim_transform_list):
        estim_transform_list[t_idx].device = torch.device('cpu')

    # Optionally visualize input scenes
    if args.vis_input_mode is not None:
        visualize_input(query_tgt, pcd_tgt, pcd_ref, args)

    # Optionally visualize local matches
    if args.vis_local_match_mode is not None:
        visualize_local_transform(query_tgt, pcd_tgt, pcd_ref, args, local_transform_list)
