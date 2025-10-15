import numpy as np
from typing import Tuple, Union
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import trimesh
from PIL import Image


def generate_uniform_query_points(num_query: Union[int, Tuple], qmin_x, qmax_x, qmin_y, qmax_y):
    if isinstance(num_query, tuple):
        qpoints_x = np.linspace(qmin_x, qmax_x, num_query[0])
        qpoints_y = np.linspace(qmin_y, qmax_y, num_query[1])
    else:
        num_query_per_axis = math.ceil(math.sqrt(num_query))
        qpoints_x = np.linspace(qmin_x, qmax_x, num_query_per_axis)
        qpoints_y = np.linspace(qmin_y, qmax_y, num_query_per_axis)
    qpoints_x, qpoints_y = np.meshgrid(qpoints_x, qpoints_y)
    qpoints = np.stack([qpoints_x, qpoints_y], axis=-1).reshape(-1, 2)
    return qpoints


def generate_grid_points(num_grid_points: int, xyz_bounds: np.ndarray, centroids: np.ndarray, rot_mtx: np.ndarray, trans_mtx: np.ndarray, scale_factors: Tuple = (1., 1., 1.)):
    # xyz_bounds is (N_obj, 1, 3), centroids is (N_obj, 3), rot_mtx is (N_obj, 3, 3), trans_mtx is (N_obj, 3), and scale_factors is a tuple with (scale_x, scale_y, scale_z)
    scale_x, scale_y, scale_z = scale_factors
    grid_points_x, grid_points_y, grid_points_z = np.meshgrid(
        np.linspace(-0.5 * scale_x, 0.5 * scale_x, num=num_grid_points),
        np.linspace(-0.5 * scale_y, 0.5 * scale_y, num=num_grid_points),
        np.linspace(-0.5 * scale_z, 0.5 * scale_z, num=num_grid_points),
        indexing='ij'
    )
    grid_points = np.stack([grid_points_x, grid_points_y, grid_points_z], axis=-1).reshape(-1, 3)
    grid_points = grid_points[None, ...] * xyz_bounds  # (N_obj, N_grid, 3)
    grid_points = grid_points + centroids[:, None, :]
    grid_points = grid_points[:, :, None, :] @ np.transpose(rot_mtx, (0, 2, 1))[:, None, :, :] + trans_mtx[:, None, None, :]
    grid_points = grid_points.squeeze(2)
    grid_points = grid_points.reshape(-1, 3)
    return grid_points


def generate_grid_points_from_centroids(num_grid_points: int, centroids: np.ndarray, scale_factors: Tuple = (1., 1., 1.)):
    # centroids is (N_c, 3), and scale_factors is a tuple with (scale_x, scale_y, scale_z)
    scale_x, scale_y, scale_z = scale_factors
    if scale_y == 0.:  # 2D case
        grid_points_x, grid_points_z = np.meshgrid(
            np.linspace(-0.5 * scale_x, 0.5 * scale_x, num=num_grid_points),
            np.linspace(-0.5 * scale_z, 0.5 * scale_z, num=num_grid_points),
            indexing='ij'
        )
        grid_points_y = np.zeros_like(grid_points_x)
    else:  # 3D case
        grid_points_x, grid_points_y, grid_points_z = np.meshgrid(
            np.linspace(-0.5 * scale_x, 0.5 * scale_x, num=num_grid_points),
            np.linspace(-0.5 * scale_y, 0.5 * scale_y, num=num_grid_points),
            np.linspace(-0.5 * scale_z, 0.5 * scale_z, num=num_grid_points),
            indexing='ij'
        )
    grid_points = np.stack([grid_points_x, grid_points_y, grid_points_z], axis=-1).reshape(-1, 3)
    grid_points = grid_points[None, ...] + centroids[:, None, :]  # (N_c, N_grid, 3)
    return grid_points


def generate_2d_grid_points(num_grid_points: int, centroids: np.ndarray, xy_bounds: np.ndarray, rot_mtx: np.ndarray, trans_mtx: np.ndarray, scale_factors: Tuple = (1., 1.)):
    # centroids is (N_obj, 2), xyz_bounds is (N_obj, 1, 2), rot_mtx is (N_obj, 2, 2), trans_mtx is (N_obj, 2), and scale_factors is a tuple with (scale_x, scale_y)
    scale_x, scale_y = scale_factors
    grid_points_x, grid_points_y = np.meshgrid(
        np.linspace(-0.5 * scale_x, 0.5 * scale_x, num=num_grid_points),
        np.linspace(-0.5 * scale_y, 0.5 * scale_y, num=num_grid_points),
        indexing='ij'
    )
    grid_points = np.stack([grid_points_x, grid_points_y], axis=-1).reshape(-1, 2)
    grid_points = grid_points[None, ...] * xy_bounds  # (N_obj, N_grid, 2)
    grid_points = grid_points + centroids[:, None, :]  # (N_obj, N_grid, 2)
    grid_points = grid_points[:, :, None, :2] @ np.transpose(rot_mtx, (0, 2, 1))[:, None, :, :] + trans_mtx[:, None, None, :]
    grid_points = grid_points.squeeze(2).reshape(-1, 2)
    return grid_points


def check_in_polygon(poly_points: np.ndarray, query_points: np.ndarray):
    # poly_points are floorplan points arranged clockwise
    valid_poly_points = poly_points[poly_points.sum(-1) != np.inf]
    poly_points_sply = Polygon(valid_poly_points)

    in_polygon = []
    for point in query_points:
        point_sply = Point(point)
        in_polygon.append(poly_points_sply.contains(point_sply))
    in_polygon = np.array(in_polygon, dtype=bool)
    return in_polygon


# Excerpted from https://github.com/nv-tlabs/ATISS
def floor_plan_from_scene(scene, path_to_floor_plan_textures, without_room_mask=False):
    floor_textures = [
        os.path.join(path_to_floor_plan_textures, fi)
        for fi in os.listdir(path_to_floor_plan_textures)
        if ('.py' not in fi) and ('.json' not in fi)  # Remove invalid files
    ]

    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)
    texture_path = os.path.join(texture, 'texture.png')

    if not os.path.exists(texture_path):
        texture_path = os.path.join(texture, 'texture.jpg')

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture_path)
        )
    )

    return tr_floor


def contour_3d_uniform_sample(contour_3d: np.ndarray, sample_step_size: float, max_points: int = 5000):
    # Assume contour_3d is a (N_contour, 3) array containing a 3D looped curve
    contour_length = np.linalg.norm(contour_3d - np.roll(contour_3d, shift=-1, axis=0), axis=-1)
    cum_contour_length = np.concatenate([np.zeros(1, ), np.cumsum(contour_length)])
    num_samples = min(int(cum_contour_length[-1] / sample_step_size), max_points)
    sample_t = cum_contour_length / cum_contour_length[-1]
    interp_t = np.linspace(0., 1., num_samples + 1)[:-1]

    aug_contour_3d = np.concatenate([contour_3d, contour_3d[0:1]], axis=0)
    interp_x = np.interp(interp_t, sample_t, aug_contour_3d[:, 0])
    interp_y = np.interp(interp_t, sample_t, aug_contour_3d[:, 1])
    interp_z = np.interp(interp_t, sample_t, aug_contour_3d[:, 2])

    interp_contour = np.stack([interp_x, interp_y, interp_z], axis=-1)
    return interp_contour


def fp_uniform_sample(fp_contour: np.ndarray, contour_step_size: float, height_step_size: float, max_height_points: int = 10, max_ground_points: int = 5000, add_floor_ceil: bool = True):
    # Asusme fp_contour is a (2 * N_contour, 3) array containing bi-level information of ground and ceiling points
    ground_level = fp_contour[:, 1].min()
    ceil_level = fp_contour[:, 1].max()
    ground_contour = fp_contour[fp_contour[:, 1] == ground_level]
    ground_samples = contour_3d_uniform_sample(ground_contour, contour_step_size, max_ground_points)

    # Add wall points
    scene_height = ceil_level - ground_level
    num_height_samples = min(int(scene_height / height_step_size), max_height_points)
    height_samples = np.linspace(ground_level, ceil_level, num_height_samples)
    wall_sample_x, wall_sample_y = np.meshgrid(ground_samples[:, 0], height_samples)
    wall_sample_z, wall_sample_y = np.meshgrid(ground_samples[:, 2], height_samples)

    wall_sample_x = wall_sample_x.reshape(-1)
    wall_sample_y = wall_sample_y.reshape(-1)
    wall_sample_z = wall_sample_z.reshape(-1)

    wall_sample_points = np.stack([wall_sample_x, wall_sample_y, wall_sample_z], axis=-1)

    # Add floor and ceiling points
    if add_floor_ceil:
        scene_min_x = fp_contour[:, 0].min()
        scene_max_x = fp_contour[:, 0].max()
        scene_min_z = fp_contour[:, 2].min()
        scene_max_z = fp_contour[:, 2].max()

        num_grid_points = (
            math.ceil((scene_max_x - scene_min_x) / contour_step_size),
            math.ceil((scene_max_z - scene_min_z) / contour_step_size)
        )
        surplus_sample_points = generate_uniform_query_points(num_grid_points, scene_min_x, scene_max_x, scene_min_z, scene_max_z)
        surplus_in_polygon = check_in_polygon(fp_contour[:fp_contour.shape[0] // 2, [0, 2]], surplus_sample_points)

        floor_sample_points = surplus_sample_points[surplus_in_polygon]
        floor_sample_points = np.stack([floor_sample_points[:, 0], np.ones_like(floor_sample_points[:, 0]) * ground_level, floor_sample_points[:, 1]], axis=-1)
        ceil_sample_points = np.copy(floor_sample_points)
        ceil_sample_points[:, 1] = ceil_level
        sample_points = np.concatenate([wall_sample_points, floor_sample_points, ceil_sample_points], axis=0)
    else:
        sample_points = wall_sample_points

    return sample_points
