import numpy as np
import networkx as nx
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from collections.abc import Callable
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_gather, knn_points, ball_query
from pytorch3d.ops.utils import masked_gather
import trimesh
from threed_front.scene_gen_utils import (
    generate_uniform_query_points,
    check_in_polygon,
    contour_3d_uniform_sample
)


def none_or_str(value):  # Type for handling both string 'None' and string
    if value == 'None':
        return None
    return value


def contour_from_floorplan_mesh(vertices: np.array, faces: np.array, simplify_contour: bool = False, simplify_angle_thres: float = 150.):
    face2edge = {}
    edge2face = {}

    unq_vertices, unq_inverse = np.unique(vertices, axis=0, return_inverse=True)

    for face in faces:
        unq_face = [unq_inverse[f] for f in face]
        face_key = tuple(unq_face)
        face2edge[face_key] = []
        for f_idx in range(3):
            edge_key = tuple(sorted([unq_face[f_idx % 3], unq_face[(f_idx + 1) % 3]]))
            face2edge[face_key].append(edge_key)

            if edge_key in edge2face.keys():
                edge2face[edge_key].append(face)
            else:
                edge2face[edge_key] = [face]

    contour_edges = []
    for edge_key in edge2face.keys():
        if len(edge2face[edge_key]) == 1:
            contour_edges.append(edge_key)

    # DFS on contour edges to get final vertices
    contour_graph = nx.Graph()
    contour_graph.add_edges_from(contour_edges)

    if simplify_contour:
        rm_node_list = []
        for node in contour_graph.nodes:  # Track nodes with degree two to remove
            if (contour_graph.degree[node] == 2) and len(list(contour_graph.neighbors(node))) > 1:  # Exclude duplicate edges to same node
                start_nbor, end_nbor = list(contour_graph.neighbors(node))
                start_pt = unq_vertices[start_nbor]
                end_pt = unq_vertices[end_nbor]
                curr_pt = unq_vertices[node]
                diff_to_start = (curr_pt - start_pt) / np.linalg.norm(curr_pt - start_pt)
                diff_to_end = (curr_pt - end_pt) / np.linalg.norm(curr_pt - end_pt)
                diff_angle = np.rad2deg(np.arccos((diff_to_start * diff_to_end).sum(axis=-1)))

                if diff_angle > simplify_angle_thres:  # Only remove large angle nodes
                    rm_node_list.append(node)

        for node in rm_node_list:  # Remove nodes and re-connect adjacent nodes
            nbors = list(contour_graph.neighbors(node))
            contour_graph.remove_node(node)

            if len(nbors) == 2:
                start_nbor, end_nbor = nbors
                contour_graph.add_edge(start_nbor, end_nbor)

        # Update graph model
        contour_graph.remove_edges_from(nx.selfloop_edges(contour_graph))  # Remove self-edges
        keep_idx = sorted(contour_graph.nodes)
        unq_vertices = unq_vertices[keep_idx]
        contour_graph = nx.relabel_nodes(contour_graph, mapping={k_idx: order_idx for (k_idx, order_idx) in zip(keep_idx, range(len(keep_idx)))})

        # Re-run DFS to get ordered set of vertices
        contour_vertices_idx = list(nx.dfs_preorder_nodes(contour_graph))
        contour_vertices = unq_vertices[contour_vertices_idx]
    else:
        contour_vertices_idx = list(nx.dfs_preorder_nodes(contour_graph))
        contour_vertices = unq_vertices[contour_vertices_idx]

    return contour_vertices


def choice_without_replacement(l: Union[List, np.array], n, return_idx=False):
    if isinstance(l, list):
        idx_list = np.random.permutation(len(l))[:n].tolist()
        choice_list = [l[idx] for idx in idx_list]

        if return_idx:
            return choice_list, idx_list
        else:
            return choice_list
    elif isinstance(l, np.ndarray):
        idx_arr = np.random.permutation(len(l))[:n]
        choice_arr = l[idx_arr]

        if return_idx:
            return choice_arr, idx_arr
        else:
            return choice_arr
    else:
        raise ValueError("Invalid input type")


def print_state(print_dict: dict):
    """
    Print current training state using values from print_dict.

    Args:
        print_dict: Dictionary containing arguments to print
    """
    print_str = ""
    for idx, key in enumerate(print_dict.keys()):
        if idx == len(print_dict.keys()) - 1:
            if type(print_dict[key]) is float:
                print_str += f"{key} = {print_dict[key]:.4f}"
            else:
                print_str += f"{key} = {print_dict[key]}"
        else:
            if type(print_dict[key]) is float:
                print_str += f"{key} = {print_dict[key]:.4f}, "
            else:
                print_str += f"{key} = {print_dict[key]}, "

    print(print_str)


def get_color_wheel() -> torch.Tensor:
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    color_wheel = torch.zeros((RY + YG + GC + CB + BM + MR, 3), dtype=torch.float32)
    counter = 0
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter += RY
    color_wheel[counter:counter + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    color_wheel[counter:counter + YG, 1] = 255
    counter += YG
    color_wheel[counter:counter + GC, 1] = 255
    color_wheel[counter:counter + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    counter += GC
    color_wheel[counter:counter + CB, 1] = 255 - torch.floor(255 * torch.arange(0, CB) / CB)
    color_wheel[counter:counter + CB, 2] = 255
    counter += CB
    color_wheel[counter:counter + BM, 2] = 255
    color_wheel[counter:counter + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    counter += BM
    color_wheel[counter:counter + MR, 2] = 255 - torch.floor(255 * torch.arange(0, MR) / MR)
    color_wheel[counter:counter + MR, 0] = 255
    return color_wheel / 255


def map_coordinates_to_color(point_cloud, color_wheel):

    xyz_max = point_cloud.max(axis=0)
    xyz_min = point_cloud.min(axis=0)
    xyz_center = (xyz_max + xyz_min) / 2

    x = point_cloud[:, 0] - xyz_min[0]
    y = point_cloud[:, 1] - xyz_min[1]
    z = point_cloud[:, 2] - xyz_min[2]

    # Calculate distances to x+z=0
    dist = np.abs(x + z) / np.sqrt(2)
    # Normalization
    dist = (dist - dist.min()) / (dist.max() - dist.min()) * 0.9 * (color_wheel.shape[0] - 1)

    # Assign colors in the color wheel
    k0 = np.floor(dist).astype(int)
    k1 = (k0 + 1) % color_wheel.shape[0]
    f = dist - k0

    # Interpolate between colors in the color wheel
    colors = (1 - f[:, None]) * color_wheel[k0] + f[:, None] * color_wheel[k1]

    return colors


class RollingSampler:  # Sample a list continually by rolling
    def __init__(self, input_list):
        self.input_list = input_list
        self.sample_start = 0
        self.sample_end = 0

    def sample(self, sample_size):
        assert sample_size <= len(self.input_list) and sample_size >= 0
        right_sample_list = self.input_list[self.sample_start: self.sample_start + sample_size]
        num_left_sample = sample_size - len(right_sample_list)
        if num_left_sample > 0:
            left_sample_list = self.input_list[:num_left_sample]
            self.sample_start = num_left_sample
        else:
            self.sample_start = self.sample_start + sample_size
            left_sample_list = []
        sample_list = right_sample_list + left_sample_list
        return sample_list


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def simsiam(p1, p2, z1, z2):
    loss_12 = -nn.functional.cosine_similarity(p1, z2.detach()).mean() / 2.
    loss_21 = -nn.functional.cosine_similarity(p2, z1.detach()).mean() / 2.

    loss = loss_12 + loss_21
    return loss


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def generate_bounding_grids(pcd: Pointclouds, num_grid_points: int, margin: float = 0., return_grid_size=False):
    device = pcd.device
    
    # Builds a bounding grid around pcd with designated margin
    pcd_bounds = pcd.get_bounding_boxes()  # (N, 3, 2)
    pcd_min = pcd_bounds[..., 0]  # (N, 3)
    pcd_max = pcd_bounds[..., 1]  # (N, 3)

    grid_min = pcd_min - margin  # (N, 3)
    grid_max = pcd_max + margin  # (N, 3)

    grid_size = (grid_max - grid_min).reshape(-1, 1, 3)  # (N, 1, 3)
    grid_center = ((grid_max + grid_min) / 2.).reshape(-1, 1, 3)  # (N, 1, 3)

    # Generate normalized grid
    grid_points_x, grid_points_y, grid_points_z = torch.meshgrid(
        torch.linspace(-0.5, 0.5, steps=num_grid_points, device=device),
        torch.linspace(-0.5, 0.5, steps=num_grid_points, device=device),
        torch.linspace(-0.5, 0.5, steps=num_grid_points, device=device),
        indexing='ij'
    )  # (N_grid, N_grid, N_grid)
    
    grid_points = torch.stack([grid_points_x, grid_points_y, grid_points_z], dim=-1).reshape(1, -1, 3)
    grid_points = grid_points * grid_size
    grid_points = grid_points + grid_center  # (N, N_grid ** 3, 3)

    grid_points = Pointclouds(grid_points)
    unit_grid_size = grid_size / (num_grid_points - 1)

    if return_grid_size:
        return grid_points, unit_grid_size
    else:
        return grid_points


def generate_random_region_2d(pcd: Pointclouds, num_grid_points: int, len_range: List[float] = [1., 2.]):
    # Generate random 2D plane regions from a batch of point clouds (assume point clouds' height values are aligned with y)
    device = pcd.device

    # Randomly sample 2D region centriods from point clouds
    pcd_points = pcd.points_padded()  # (N, N_pcd, 3)
    centroids_idx = torch.randint(low=0, high=pcd_points.shape[1] + 1, size=[pcd_points.shape[0], 1, 1], device=device)
    centroids = torch.take_along_dim(pcd_points, centroids_idx, dim=1)  # (N, 1, 3)

    # Generate normalized grid
    grid_points_x, grid_points_z = torch.meshgrid(
        torch.linspace(-0.5, 0.5, steps=num_grid_points, device=device),
        torch.linspace(-0.5, 0.5, steps=num_grid_points, device=device),
        indexing='ij'
    )  # (N_grid, N_grid)
    grid_points_y = torch.zeros_like(grid_points_x)
    grid_size = torch.rand([centroids.shape[0], 1, 3], device=device) * (len_range[1] - len_range[0]) + len_range[0]
    grid_size[..., 1] = 0.  # Set y-axis grid sizes to zero
    
    grid_points = torch.stack([grid_points_x, grid_points_y, grid_points_z], dim=-1).reshape(1, -1, 3)
    grid_points = grid_points * grid_size
    grid_points = grid_points + centroids  # (N, N_grid ** 2, 3)

    grid_points = Pointclouds(grid_points)
    return grid_points


def generate_centroid_region_2d(points: torch.Tensor, num_grid_points: int, grid_size_xz: List[float] = [1., 2.], height_fix_margin: float = None):
    # Generate centroid from point cloud bounding box
    centroids = (points.max(dim=0, keepdims=True).values + points.min(dim=0, keepdims=True).values) / 2.

    if height_fix_margin is not None:  # Fix height of 2D region by adding a margin
        centroids[:, 1] = points.min(dim=0).values[1] + height_fix_margin

    # Generate normalized grid
    grid_points_x, grid_points_z = torch.meshgrid(
        torch.linspace(-0.5, 0.5, steps=num_grid_points),
        torch.linspace(-0.5, 0.5, steps=num_grid_points),
        indexing='ij'
    )  # (N_grid, N_grid)
    grid_points_y = torch.zeros_like(grid_points_x)
    grid_points = torch.stack([grid_points_x, grid_points_y, grid_points_z], dim=-1)
    grid_points[..., 0] = grid_points[..., 0] * grid_size_xz[0]
    grid_points[..., 2] = grid_points[..., 2] * grid_size_xz[1]
    grid_points = grid_points + centroids  # (N_grid ** 2, 3)

    return grid_points


def box_align(pcd_src: torch.Tensor, pcd_tgt: torch.Tensor):
    # Aligns point clouds in pcd_src of shape (N_src, 3) to pcd_tgt of shape (N_tgt, 3)
    bbox_src = torch.cat([pcd_src.min(0, keepdims=True).values, pcd_src.max(0, keepdims=True).values], dim=0)  # (2, 3)
    centroid_src = bbox_src.mean(dim=0)  # (1, 3)
    scale_src = bbox_src[0:1] - bbox_src[1:2]  # (1, 3)

    bbox_tgt = torch.cat([pcd_tgt.min(0, keepdims=True).values, pcd_tgt.max(0, keepdims=True).values], dim=0)  # (2, 3)
    centroid_tgt = bbox_tgt.mean(dim=0)  # (1, 3)
    scale_tgt = bbox_tgt[0:1] - bbox_tgt[1:2]  # (1, 3)

    scale_transform = scale_tgt / scale_src
    pcd_transform = scale_transform * (pcd_src - centroid_src) + centroid_tgt

    return pcd_transform


def box_align_2d(pcd_src: torch.Tensor, pcd_tgt: torch.Tensor):
    # Aligns point clouds in pcd_src of shape (N_src, 3) to pcd_tgt of shape (N_tgt, 3) only using xz-coordinates (assuming height is aligned)
    bbox_src = torch.cat([pcd_src.min(0, keepdims=True).values, pcd_src.max(0, keepdims=True).values], dim=0)  # (2, 3)
    centroid_src = bbox_src.mean(dim=0, keepdims=True)  # (1, 3)
    scale_src = bbox_src[0:1] - bbox_src[1:2]  # (1, 3)

    bbox_tgt = torch.cat([pcd_tgt.min(0, keepdims=True).values, pcd_tgt.max(0, keepdims=True).values], dim=0)  # (2, 3)
    centroid_tgt = bbox_tgt.mean(dim=0, keepdims=True)  # (1, 3)
    scale_tgt = bbox_tgt[0:1] - bbox_tgt[1:2]  # (1, 3)

    # Ignore height-direction values
    centroid_tgt[:, 1] = centroid_src[:, 1]
    scale_tgt[:, 1] = 1.
    scale_src[:, 1] = 1.

    scale_transform = scale_tgt / scale_src
    pcd_transform = scale_transform * (pcd_src - centroid_src) + centroid_tgt

    return pcd_transform


# Optimal transport functions excerpted from https://github.com/magicleap/SuperGluePretrainedNetwork
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, iters: int, use_dustbin: bool = False) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    if use_dustbin:
        alpha = torch.tensor(1., device=scores.device)
        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                            torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    else:
        alpha = torch.tensor(1., device=scores.device)
        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = scores
        norm = - (ms + ns).log()
        log_mu = norm.expand(m)
        log_nu = norm.expand(n)
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def apply_yaw_perturbation(coords: torch.Tensor, perturb_size: float):
    # coords is tensor of shape (N, P, 3) and perturb_size is in radians
    N, P = coords.shape[:2]
    center = coords.mean(dim=1, keepdim=True)  # (N, 1, 3)
    mod_coords = coords - center  # (N, P, 3)
    random_yaw = torch.ones(N, 1, 1, device=coords.device) * perturb_size
    random_rot = torch.cat([torch.cat([torch.cos(random_yaw), -torch.sin(random_yaw)], dim=-1),
        torch.cat([torch.sin(random_yaw), torch.cos(random_yaw)], dim=-1)], dim=1)  # (N, 2, 2)
    mod_coords[:, :, [0, 2]] = mod_coords[:, :, [0, 2]] @ torch.transpose(random_rot, 1, 2)
    mod_coords = mod_coords + center
    return mod_coords


def farthest_point_down_sample(points_np: np.array, n, return_idx=False, max_input_size=10000):
    if max_input_size != -1:  # First random sample to designated size
        in_points_np, in_idx_np = choice_without_replacement(points_np, max_input_size, return_idx=True)
    else:
        in_points_np, in_idx_np = points_np, np.arange(points_np.shape[0])

    pcd = o3d.geometry.PointCloud()
    in_idx_np = np.stack([in_idx_np.astype(float)] * 3, axis=-1)  # (N, 3)
    pcd.points = o3d.utility.Vector3dVector(in_points_np)
    pcd.colors = o3d.utility.Vector3dVector(in_idx_np)
    pcd = pcd.farthest_point_down_sample(n)

    if return_idx:
        idx_np = np.asarray(pcd.colors)[:, 0].astype(int).tolist()
        return np.asarray(pcd.points), idx_np
    else:
        return np.asarray(pcd.points)


def trimesh_load_with_postprocess(mesh_path, postprocess_type=None):
    tr_mesh = trimesh.load(mesh_path, force="mesh")
    if postprocess_type == 'bottom_crop':
        plane_origin = np.zeros([3, ])
        plane_origin[0] = tr_mesh.vertices.mean(0)[0]
        plane_origin[1] = tr_mesh.vertices.min(axis=0)[1]
        plane_origin[2] = tr_mesh.vertices.mean(0)[2]
        processed_tr_mesh = tr_mesh.slice_plane(plane_origin=plane_origin.tolist(), plane_normal=[0., 1., 0.])
    else:
        processed_tr_mesh = tr_mesh

    return processed_tr_mesh


def generate_yaw_points(num_rot: int, device='cpu'):
    yaw_arr = torch.arange(num_rot, dtype=torch.float, device=device)
    yaw_arr = yaw_arr * 2 * np.pi / num_rot

    return yaw_arr


def yaw2rot_mtx(yaw_arr: torch.Tensor, apply_xz_flip=False):
    # Initialize rotation matrices from yaw values
    def _yaw2mtx(yaw):
        # yaw is assumed to be a scalar
        yaw = yaw.reshape(1, )

        tensor_0 = torch.zeros(1, device=yaw.device)
        tensor_1 = torch.ones(1, device=yaw.device)

        R = torch.stack([
            torch.stack([torch.cos(yaw), tensor_0, -torch.sin(yaw)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([torch.sin(yaw), tensor_0, torch.cos(yaw)])
        ]).reshape(3, 3)

        return R

    tot_mtx = []
    for yaw in yaw_arr:
        if apply_xz_flip:
            tot_mtx.append(_yaw2mtx(yaw))
            tot_mtx.append(_yaw2mtx(yaw) @ np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))  # X flip (Z flip is subsumed by X flip + rotation)
        else:
            tot_mtx.append(_yaw2mtx(yaw))
    return torch.stack(tot_mtx)


def rot_mtx2yaw(rot_mtx):
    def _mtx2yaw(in_mtx):
        # in_mtx is assumed to have a shape of [3, 3]
        yaw = torch.atan2(in_mtx[2, 0], in_mtx[0, 0] + 1e-6)
        return yaw

    if len(rot_mtx.shape) == 2:
        return _mtx2yaw(rot_mtx).reshape(-1, )
    else:
        tot_mtx = []
        for mtx in rot_mtx:
            tot_mtx.append(_mtx2yaw(mtx))
        return torch.cat(tot_mtx)


def idw_nn_interpolation(interp_x: torch.Tensor, data_x: torch.Tensor, data_y: torch.Tensor, dist_pow: int = 1., agg_k: int = 5):
    # interp_x, data_x are of shape (N_interp, 3) and (N_data, 3) while data_y is of shape (N_data, D_y)
    eps = 1e-6
    N_interp = interp_x.shape[0]
    D_y = data_y.shape[1]

    interp_x_pcd = Pointclouds(interp_x[None, ...])
    data_pcd = Pointclouds(points=data_x[None, ...], features=data_y[None, ...])
    agg_results = knn_points(
        interp_x_pcd.points_padded(),
        data_pcd.points_padded(),
        lengths1=interp_x_pcd.num_points_per_cloud(),
        lengths2=data_pcd.num_points_per_cloud(),
        K=agg_k
    )
    agg_dists = torch.sqrt(agg_results.dists)  # (1, N_interp, K)
    agg_idx = agg_results.idx  # (1, N_interp, K)
    agg_y = knn_gather(
        data_pcd.features_padded(),
        agg_idx,
        data_pcd.num_points_per_cloud()
    )  # (1, N_interp, K, D_y)

    agg_dists, agg_y = agg_dists.reshape(N_interp, agg_k, 1), agg_y.reshape(N_interp, agg_k, D_y)  # Remove batch dimension

    inv_dist = 1 / torch.pow(agg_dists + eps, dist_pow)  # (N_interp, K, 1)
    normalizer = inv_dist.sum(dim=1, keepdim=True)  # (N_interp, 1, 1)
    interp_y = (agg_y * inv_dist) / normalizer  # (N_interp, K, D_y)
    interp_y = interp_y.sum(dim=1)  # (N_interp, D_y)

    return interp_y


def radial_idw_nn_interpolation(interp_x: torch.Tensor, data_x: torch.Tensor, data_y: torch.Tensor, dist_pow: int = 1., agg_radius: float = 1., adaptive_k_func: Callable = None, agg_radius_limit: int = 300):
    # interp_x, data_x are of shape (N_interp, 3) and (N_data, 3) while data_y is of shape (N_data, D_y)
    eps = 1e-6
    N_interp = interp_x.shape[0]
    D_y = data_y.shape[1]

    if adaptive_k_func is not None:
        adaptive_k = adaptive_k_func(agg_radius)
    else:
        adaptive_k = 500

    adaptive_k = min(data_x.shape[0], adaptive_k)  # adaptive_k does not need to exceed maximum number of points

    interp_x_pcd = Pointclouds(interp_x[None, ...])
    data_pcd = Pointclouds(points=data_x[None, ...], features=data_y[None, ...])
    agg_results = ball_query(
        interp_x_pcd.points_padded(),
        data_pcd.points_padded(),
        lengths1=interp_x_pcd.num_points_per_cloud(),
        lengths2=data_pcd.num_points_per_cloud(),
        K=adaptive_k,
        radius=agg_radius,
        return_nn=True
    )
    agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
    agg_idx = agg_results.idx  # (N_batch, N_query, K)

    # Ensure maximum number of points to be aggregated via random sampling
    if adaptive_k > agg_radius_limit:
        sampling_weight = (agg_dists[0] != 0.).float()
        sampling_weight[sampling_weight.sum(dim=-1) == 0.] += 1.  # Ensure all weights sum to non-zero
        sampling_idx = torch.multinomial(sampling_weight, agg_radius_limit, replacement=False)
        agg_dists = torch.take_along_dim(agg_dists[0], sampling_idx, dim=-1).reshape(1, N_interp, -1, 1)
        agg_idx = torch.take_along_dim(agg_idx[0], sampling_idx, dim=-1).reshape(1, N_interp, -1)
        adaptive_k = agg_radius_limit

    agg_y = masked_gather(data_pcd.features_padded(), agg_idx)  # (N_batch, N_query, K, D_y)

    agg_dists, agg_y = agg_dists.reshape(N_interp, adaptive_k, 1), agg_y.reshape(N_interp, adaptive_k, D_y)  # Remove batch dimension

    invalid_entry_mask = (agg_idx == -1).reshape(-1, adaptive_k)  # (N_batch * N_query, K)
    agg_dists[invalid_entry_mask] = torch.inf  # Infinity values for zeroing out distance weight matrix

    inv_dist = 1 / torch.pow(agg_dists + eps, dist_pow)  # (N_interp, K, 1)
    normalizer = inv_dist.sum(dim=1, keepdim=True)  # (N_interp, 1, 1)
    interp_y = (agg_y * inv_dist) / (normalizer + eps)  # (N_interp, K, D_y)
    interp_y = interp_y.sum(dim=1)  # (N_interp, D_y)

    return interp_y


def dist_softmax_weighted_average(interp_x: torch.Tensor, data_x: torch.Tensor, data_y: torch.Tensor, agg_exp_temp: int = 1., agg_k: int = 5):
    # interp_x, data_x are of shape (N_interp, 3) and (N_data, 3) while data_y is of shape (N_data, D_y)
    eps = 1e-6
    N_interp = interp_x.shape[0]
    D_y = data_y.shape[1]

    interp_x_pcd = Pointclouds(interp_x[None, ...])
    data_pcd = Pointclouds(points=data_x[None, ...], features=data_y[None, ...])
    agg_results = knn_points(
        interp_x_pcd.points_padded(),
        data_pcd.points_padded(),
        lengths1=interp_x_pcd.num_points_per_cloud(),
        lengths2=data_pcd.num_points_per_cloud(),
        K=agg_k
    )
    agg_dists = torch.sqrt(agg_results.dists)  # (1, N_interp, K)
    agg_idx = agg_results.idx  # (1, N_interp, K)
    agg_y = knn_gather(
        data_pcd.features_padded(),
        agg_idx,
        data_pcd.num_points_per_cloud()
    )  # (1, N_interp, K, D_y)
    agg_dists_flatten = agg_dists.reshape(-1, agg_k, 1)

    invalid_entry_mask = (agg_idx == -1).reshape(-1, agg_k)  # (1 * N_interp, K)
    agg_dists_flatten[invalid_entry_mask] = torch.inf  # Infinity values for zeroing out distance weight matrix

    dist_softmax_mtx = torch.softmax(-agg_dists_flatten / agg_exp_temp, dim=1)  # (1 * N_interp, K, 1)
    dist_softmax_mtx[torch.isnan(dist_softmax_mtx)] = 0.  # Zero out NaN values as they will all be set to zero from exponentials
    dist_wgt_mtx = dist_softmax_mtx * torch.exp(-agg_dists_flatten / agg_exp_temp)  # (1 * N_interp, K, 1)

    agg_y = agg_y.reshape(N_interp, agg_k, D_y)  # Remove batch dimension
    interp_y = (dist_wgt_mtx * agg_y).mean(dim=1)  # (N_interp, D_y)

    return interp_y


def radial_dist_softmax_weighted_average(interp_x: torch.Tensor, data_x: torch.Tensor, data_y: torch.Tensor, agg_exp_temp: int = 1., agg_radius: float = 1., adaptive_k_func: Callable = None, agg_radius_limit: int = 300):
    # interp_x, data_x are of shape (N_interp, 3) and (N_data, 3) while data_y is of shape (N_data, D_y)
    eps = 1e-6
    N_interp = interp_x.shape[0]
    D_y = data_y.shape[1]

    if adaptive_k_func is not None:
        adaptive_k = adaptive_k_func(agg_radius)
    else:
        adaptive_k = 500

    adaptive_k = min(data_x.shape[0], adaptive_k)  # adaptive_k does not need to exceed maximum number of points

    interp_x_pcd = Pointclouds(interp_x[None, ...])
    data_pcd = Pointclouds(points=data_x[None, ...], features=data_y[None, ...])
    agg_results = ball_query(
        interp_x_pcd.points_padded(),
        data_pcd.points_padded(),
        lengths1=interp_x_pcd.num_points_per_cloud(),
        lengths2=data_pcd.num_points_per_cloud(),
        K=adaptive_k,
        radius=agg_radius,
        return_nn=True
    )
    agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
    agg_idx = agg_results.idx  # (N_batch, N_query, K)

    # Ensure maximum number of points to be aggregated via random sampling
    if adaptive_k > agg_radius_limit:
        sampling_weight = (agg_dists[0] != 0.).float()
        sampling_weight[sampling_weight.sum(dim=-1) == 0.] += 1.  # Ensure all weights sum to non-zero
        sampling_idx = torch.multinomial(sampling_weight, agg_radius_limit, replacement=False)
        agg_dists = torch.take_along_dim(agg_dists[0], sampling_idx, dim=-1).reshape(1, N_interp, -1, 1)
        agg_idx = torch.take_along_dim(agg_idx[0], sampling_idx, dim=-1).reshape(1, N_interp, -1)
        adaptive_k = agg_radius_limit

    agg_y = masked_gather(data_pcd.features_padded(), agg_idx)  # (N_batch, N_query, K, D_y)
    agg_dists_flatten = agg_dists.reshape(-1, adaptive_k, 1)

    invalid_entry_mask = (agg_idx == -1).reshape(-1, adaptive_k)  # (1 * N_interp, K)
    agg_dists_flatten[invalid_entry_mask] = torch.inf  # Infinity values for zeroing out distance weight matrix

    dist_softmax_mtx = torch.softmax(-agg_dists_flatten / agg_exp_temp, dim=1)  # (1 * N_interp, K, 1)
    dist_softmax_mtx[torch.isnan(dist_softmax_mtx)] = 0.  # Zero out NaN values as they will all be set to zero from exponentials
    dist_wgt_mtx = dist_softmax_mtx * torch.exp(-agg_dists_flatten / agg_exp_temp)  # (1 * N_interp, K, 1)

    agg_y = agg_y.reshape(N_interp, adaptive_k, D_y)  # Remove batch dimension
    interp_y = (dist_wgt_mtx * agg_y).mean(dim=1)  # (N_interp, D_y)

    return interp_y


def mutual_nn_pairs(pcd0, pcd1, nn_dist_thres=None):
    pcd0_np = np.asarray(pcd0.points)
    pcd1_np = np.asarray(pcd1.points)
    pcd0_range = np.arange(pcd0_np.shape[0])
    pcd1_range = np.arange(pcd1_np.shape[0])
    dist_mtx = np.linalg.norm(pcd0_np[:, None, :] - pcd1_np[None, :, :], axis=-1)  # (N_0, N_1)

    # Mutual NN assignment (https://gist.github.com/mihaidusmanu/20fd0904b2102acc1330bad9b4badab8)
    match_0_to_1 = dist_mtx.argmin(-1)  # (N_0)
    match_1_to_0 = dist_mtx.argmin(0)  # (N_1)

    if nn_dist_thres is not None:
        valid_matches = (match_1_to_0[match_0_to_1] == pcd0_range) & (dist_mtx.min(-1) < nn_dist_thres)
    else:
        valid_matches = match_1_to_0[match_0_to_1] == pcd0_range

    match_0_idx = pcd0_range[valid_matches]
    match_1_idx = pcd1_range[match_0_to_1[valid_matches]]

    match_pcd0 = o3d.geometry.PointCloud()
    match_pcd0_np = pcd0_np[match_0_idx]
    match_pcd0.points = o3d.utility.Vector3dVector(match_pcd0_np)

    match_pcd1 = o3d.geometry.PointCloud()
    match_pcd1_np = pcd1_np[match_1_idx]
    match_pcd1.points = o3d.utility.Vector3dVector(match_pcd1_np)

    return match_pcd0, match_0_idx, match_pcd1, match_1_idx


def o3d_geometry_copy(in_geometry):
    if isinstance(in_geometry, o3d.geometry.TriangleMesh):
        return o3d.geometry.TriangleMesh(in_geometry)
    if isinstance(in_geometry, o3d.geometry.PointCloud):
        return o3d.geometry.PointCloud(in_geometry)
    if isinstance(in_geometry, o3d.geometry.LineSet):
        return o3d.geometry.LineSet(in_geometry)
    if isinstance(in_geometry, o3d.geometry.AxisAlignedBoundingBox):
        return o3d.geometry.AxisAlignedBoundingBox(in_geometry)
    if isinstance(in_geometry, o3d.geometry.OrientedBoundingBox):
        return o3d.geometry.OrientedBoundingBox(in_geometry)


def o3d_geometry_list_aabb(in_geometry_list: List):
    result_aabb = np.ones([2, 3])  # (Max, Min) 3D coordinates
    result_aabb[0, :] *= -np.inf  # Max bounds
    result_aabb[1, :] *= np.inf  # Min bounds
    for in_geometry in in_geometry_list:
        bounds = in_geometry.get_axis_aligned_bounding_box()
        result_aabb[0, :] = np.maximum(bounds.max_bound, result_aabb[0, :])
        result_aabb[1, :] = np.minimum(bounds.min_bound, result_aabb[1, :])
    return result_aabb


def o3d_geometry_list_shift(in_geometry_list: List, shift_amount: List = [0., 0., 0.]):
    out_geometry_list = []
    for in_geometry in in_geometry_list:
        if isinstance(in_geometry, o3d.geometry.TriangleMesh):
            out_geometry = o3d.geometry.TriangleMesh(in_geometry)
            out_geometry.translate(np.array([shift_amount], dtype=np.float64).T)
        if isinstance(in_geometry, o3d.geometry.PointCloud):
            out_geometry = o3d.geometry.PointCloud(in_geometry)
            out_geometry.translate(np.array([shift_amount], dtype=np.float64).T)
        if isinstance(in_geometry, o3d.geometry.LineSet):
            out_geometry = o3d.geometry.LineSet(in_geometry)
            out_geometry.translate(np.array([shift_amount], dtype=np.float64).T)
        if isinstance(in_geometry, o3d.geometry.AxisAlignedBoundingBox):
            out_geometry = o3d.geometry.AxisAlignedBoundingBox(in_geometry)
            out_geometry.translate(np.array([shift_amount], dtype=np.float64).T)
        if isinstance(in_geometry, o3d.geometry.OrientedBoundingBox):
            out_geometry = o3d.geometry.OrientedBoundingBox(in_geometry)
            out_geometry.translate(np.array([shift_amount], dtype=np.float64).T)
        out_geometry_list.append(out_geometry)
    return out_geometry_list


def o3d_geometry_list_scale(in_geometry_list: List, scale_amount: float = 1., centroid: np.ndarray = np.zeros([3, 1])):
    out_geometry_list = []
    for in_geometry in in_geometry_list:
        if isinstance(in_geometry, o3d.geometry.TriangleMesh):
            out_geometry = o3d.geometry.TriangleMesh(in_geometry)
            out_geometry.scale(scale_amount, centroid)
        if isinstance(in_geometry, o3d.geometry.PointCloud):
            out_geometry = o3d.geometry.PointCloud(in_geometry)
            out_geometry.scale(scale_amount, centroid)
        if isinstance(in_geometry, o3d.geometry.LineSet):
            out_geometry = o3d.geometry.LineSet(in_geometry)
            out_geometry.scale(scale_amount, centroid)
        if isinstance(in_geometry, o3d.geometry.AxisAlignedBoundingBox):
            out_geometry = o3d.geometry.AxisAlignedBoundingBox(in_geometry)
            out_geometry.scale(scale_amount, centroid)
        if isinstance(in_geometry, o3d.geometry.OrientedBoundingBox):
            out_geometry = o3d.geometry.OrientedBoundingBox(in_geometry)
            out_geometry.scale(scale_amount, centroid)
        out_geometry_list.append(out_geometry)
    return out_geometry_list


def keypoints_to_spheres(keypoints, paint_color=None, radius=0.015, alpha=1.0):
    spheres = o3d.geometry.TriangleMesh()
    for point_idx, alphapoint in enumerate(keypoints.points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(alphapoint)
        if len(keypoints.colors) != 0:
            sphere_colors_np = keypoints.colors[point_idx][None, :].repeat(len(sphere.vertices), axis=0)
            sphere.vertex_colors = o3d.utility.Vector3dVector(sphere_colors_np)
        spheres += sphere
    if paint_color is not None:
        spheres.paint_uniform_color(paint_color)
    return spheres


def str2cls_3dfuture(s, return_type='one_hot'):
    str_list = [
        'bed',  # 0
        'pier/stool',  # 1
        'others',  # 2
        'sofa',  # 3
        'chair',  # 4
        'lighting',  # 5
        'table',  # 6
        'cabinet/shelf/desk'  # 7
    ]
    str2cls_dict = {s: s_idx for s_idx, s in enumerate(str_list)}
    if return_type == 'one_hot':
        return np.eye(len(str_list))[str2cls_dict[s]]
    else:
        return str2cls_dict[s]


class CollisionFreePlacer:
    """
        Class for calculating 2D discrete locations for placing objects. Proceeds through a three-step process.

        1. Uniformly sample points within floorplan interior.
        2. Remove point locations covered by occupied regions.
        3. Placement locations are updated after placing new objects.
    """
    def __init__(self, fp_points: np.ndarray, num_search_points: int, occ_centroids: np.ndarray, occ_radii: np.ndarray):
        # NOTE: fp_points, occ_centroids are all (N, 3) arrays, while we only use 0-th and 2-nd components for computing placements
        # Asusme fp_points is a (2 * N_contour, 3) array containing bi-level information of ground and ceiling points
        # Generate points within floorplan interior

        if len(fp_points) == 0:  # Make a bounding floorplan if no fp_points exist
            self.placeable = False
            print("Invalid floorplan: aborting!")
            return

        scene_min_x = fp_points[:, 0].min()
        scene_max_x = fp_points[:, 0].max()
        scene_min_z = fp_points[:, 2].min()
        scene_max_z = fp_points[:, 2].max()
        init_num_query = 1000  # Used for determining area ratio between floorplan & bounding box
        init_query_points = generate_uniform_query_points(init_num_query, scene_min_x, scene_max_x, scene_min_z, scene_max_z)

        init_in_polygon = check_in_polygon(fp_points[:fp_points.shape[0] // 2, [0, 2]], init_query_points)
        area_ratio = init_in_polygon.sum() / init_in_polygon.shape[0]
        if len(init_query_points) == 0 or area_ratio == 0.:  # Stop initialization if there's an invalid floorplan with no available interior points
            self.placeable = False
            print("Invalid floorplan: aborting!")
            return

        num_surplus = num_search_points / area_ratio  # Discount for area_ratio when generating query points
        surplus_points = generate_uniform_query_points(num_surplus, scene_min_x, scene_max_x, scene_min_z, scene_max_z)
        surplus_in_polygon = check_in_polygon(fp_points[:fp_points.shape[0] // 2, [0, 2]], surplus_points)

        self.sample_points = surplus_points[surplus_in_polygon]
        self.occ_centroids = occ_centroids[:, [0, 2]]
        self.occ_radii = occ_radii
        self.num_occ = occ_radii.shape[0]
        self.placeable = True  # Flag determining if there are remaining sample points to place

        # Carve out points that are occupied
        for occ_idx in range(self.num_occ):
            centroid_dist = np.linalg.norm(self.sample_points - self.occ_centroids[occ_idx: occ_idx + 1], axis=-1)
            self.sample_points = self.sample_points[centroid_dist > self.occ_radii[occ_idx]]

            if len(self.sample_points) == 0:
                self.placeable = False
                break

        # Cache floorplan contour points
        self.fp_contour = contour_3d_uniform_sample(fp_points[:fp_points.shape[0] // 2], 0.3, max_points=100)[:, [0, 2]]

    def update_placement(self, new_centroid: np.ndarray, new_radius: np.ndarray):  # Update placeable regions after new object addition
        # new_centroid is assumed to either have shape (3, ) for 3D inputs and (2, ) for 2D inputs and new_radius is assumed to have shape (1, )
        if len(new_centroid) == 2:
            new_centroid = np.array([new_centroid[0], 0., new_centroid[1]])  # Modify to 3D array

        centroid_dist = np.linalg.norm(self.sample_points - new_centroid[None, [0, 2]], axis=-1)
        self.sample_points = self.sample_points[centroid_dist > new_radius]

        if len(self.sample_points) == 0:
            self.placeable = False

        self.num_occ += 1
        self.occ_centroids = np.concatenate([self.occ_centroids, new_centroid[None, [0, 2]]], axis=0)
        self.occ_radii = np.concatenate([self.occ_radii, np.ones([1, ]) * new_radius], axis=0)

    def compute_feasibility(self, new_radius: np.ndarray):  # Compute whether an object with new_radius is placeable
        # new_centroid is assumed to have shape (3, ) and new_radius is assumed to have shape (1, )
        # Track locations not colliding with floorplan contours
        contour_dist = np.linalg.norm(self.sample_points[:, None, :] - self.fp_contour[None, :, :], axis=-1).min(axis=-1)
        feasible_points = self.sample_points[(contour_dist > new_radius)]

        # Track locations not colliding with objects
        for occ_idx in range(self.num_occ):
            centroid_dist = np.linalg.norm(feasible_points - self.occ_centroids[occ_idx: occ_idx + 1], axis=-1)
            feasible_points = feasible_points[centroid_dist > self.occ_radii[occ_idx] + new_radius]

            if len(feasible_points) == 0:
                placeable = False
                return feasible_points, placeable

        placeable = (len(feasible_points) != 0)
        return feasible_points, placeable

