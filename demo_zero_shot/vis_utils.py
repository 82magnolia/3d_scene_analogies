import numpy as np
import open3d as o3d
from pytorch3d.structures import Pointclouds
from typing import NamedTuple, List
from threed_front.utils import (
    get_color_wheel,
    map_coordinates_to_color,
    keypoints_to_spheres,
    o3d_geometry_list_aabb,
    o3d_geometry_list_scale,
    o3d_geometry_list_shift,
    generate_random_region_2d
)

# Ideal scene bounding box size for visualization
IDEAL_VIS_LENGTH = np.array([12.0, 3.2, 12.0])


def visualize_input(query_tgt: Pointclouds, pcd_tgt: o3d.geometry.PointCloud, pcd_ref: o3d.geometry.PointCloud, args: NamedTuple):
    # NOTE: We assume a single pair of scenes as input
    vis_query_tgt = query_tgt.points_list()[0].cpu()

    # Make color maps similar to dense semantic flow methods (https://github.com/kampta/asic)
    color_wheel = get_color_wheel().numpy()
    idx_color = map_coordinates_to_color(np.array(vis_query_tgt), color_wheel)

    # Make final visualization targets
    vis_tgt = np.concatenate([vis_query_tgt, idx_color], axis=-1)  # (N_query, 6)
    vis_tgt_pcd = o3d.geometry.PointCloud()
    vis_tgt_pcd.points = o3d.utility.Vector3dVector(vis_tgt[:, :3])
    vis_tgt_pcd.colors = o3d.utility.Vector3dVector(vis_tgt[:, 3:])
    vis_tgt_pcd = keypoints_to_spheres(vis_tgt_pcd, radius=0.1)

    vis_tgt_scene = [pcd_tgt] + [vis_tgt_pcd]
    vis_ref_scene = [pcd_ref]

    # Compute scale amounts for decent visualization
    fp_tgt_bounds = o3d_geometry_list_aabb([pcd_tgt])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
    fp_ref_bounds = o3d_geometry_list_aabb([pcd_ref])

    fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
    fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

    resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
    resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
    vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
    vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

    # Compute shift amounts from bounding box
    fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len([pcd_tgt])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
    fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len([pcd_ref])])

    vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
    vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
    vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
    vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

    vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
    vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

    # Fix both scenes' ground level
    tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len([pcd_tgt])])[1, 1]
    ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len([pcd_ref])])[1, 1]

    vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
    vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
    geometry_list = vis_tgt_scene + vis_ref_scene

    o3d.visualization.draw_geometries(geometry_list)


def visualize_local_transform(query_tgt: Pointclouds, pcd_tgt: o3d.geometry.PointCloud, pcd_ref: o3d.geometry.PointCloud, args: NamedTuple, local_transform_list: List):
    # NOTE: We assume a single pair of scenes as input
    query_texture = [feat[..., 1:] for feat in query_tgt.features_list()]

    # Features are offloaded to CPU to save GPU memory
    vis_query_tgt = query_tgt.points_list()[0].cpu()
    vis_texture = query_texture[0].cpu()

    # Make color maps similar to dense semantic flow methods (https://github.com/kampta/asic)
    color_wheel = get_color_wheel().numpy()
    idx_color = map_coordinates_to_color(np.array(vis_query_tgt), color_wheel)

    # Visualize initial target points if there are no local transforms available
    if len(local_transform_list[0]) == 0:
        visualize_input(query_tgt, pcd_tgt, pcd_ref, args)

    for t_idx, transform in enumerate(local_transform_list[0]):
        vis_deform_query_tgt = transform(vis_query_tgt)

        if args.vis_local_match_mode in ['intra_match', 'texture_transfer']:
            if args.vis_local_match_mode == 'intra_match':
                vis_tgt = np.concatenate([vis_query_tgt, idx_color], axis=-1)  # (N_query, 6)
                vis_deform_tgt = np.concatenate([vis_deform_query_tgt, idx_color], axis=-1)  # (N_query, 6)
            else:
                vis_tgt = np.concatenate([vis_query_tgt, vis_texture], axis=-1)  # (N_query, 6)
                vis_deform_tgt = np.concatenate([vis_deform_query_tgt, vis_texture], axis=-1)  # (N_query, 6)

            vis_tgt_pcd_o3d = o3d.geometry.PointCloud()
            vis_tgt_pcd_o3d.points = o3d.utility.Vector3dVector(vis_tgt[:, :3])
            vis_tgt_pcd_o3d.colors = o3d.utility.Vector3dVector(vis_tgt[:, 3:])
            vis_deform_tgt_pcd_o3d = o3d.geometry.PointCloud()
            vis_deform_tgt_pcd_o3d.points = o3d.utility.Vector3dVector(vis_deform_tgt[:, :3])
            vis_deform_tgt_pcd_o3d.colors = o3d.utility.Vector3dVector(vis_deform_tgt[:, 3:])

            vis_tgt_pcd = keypoints_to_spheres(vis_tgt_pcd_o3d, radius=0.05)
            vis_deform_tgt_pcd = keypoints_to_spheres(vis_deform_tgt_pcd_o3d, radius=0.05)

            vis_tgt_scene = [pcd_tgt] + [vis_tgt_pcd]
            vis_ref_scene = [pcd_ref] + [vis_deform_tgt_pcd]

            # Compute scale amounts for decent visualization
            fp_tgt_bounds = o3d_geometry_list_aabb([pcd_tgt])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
            fp_ref_bounds = o3d_geometry_list_aabb([pcd_ref])

            fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
            fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

            resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
            resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
            vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
            vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

            # Compute shift amounts from bounding box
            fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len([pcd_tgt])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
            fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len([pcd_ref])])

            vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
            vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
            vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
            vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

            vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
            vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

            # Fix both scenes' ground level
            tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len([pcd_tgt])])[1, 1]
            ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len([pcd_ref])])[1, 1]

            vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
            vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
            geometry_list = vis_tgt_scene + vis_ref_scene

            o3d.visualization.draw_geometries(geometry_list)

        elif args.vis_local_match_mode in ['region_2d_match', 'region_2d_match_overlay']:
            if t_idx == 0:  # Set tgt_region_2d fixed during visualization
                tgt_region_2d = generate_random_region_2d(query_tgt, num_grid_points=args.vis_region_2d_num_grid, len_range=args.vis_region_2d_len_range)
                vis_region_2d_tgt = tgt_region_2d.points_padded()[0].cpu()
            vis_deform_region_2d_tgt = transform(vis_region_2d_tgt)
            region_2d_idx_color = map_coordinates_to_color(np.array(vis_region_2d_tgt), color_wheel)

            vis_tgt = np.concatenate([vis_region_2d_tgt.cpu(), region_2d_idx_color], axis=-1)  # (N_query, 6)
            vis_deform_tgt = np.concatenate([vis_deform_region_2d_tgt.cpu(), region_2d_idx_color], axis=-1)  # (N_query, 6)

            vis_tgt_pcd = o3d.geometry.PointCloud()
            vis_tgt_pcd.points = o3d.utility.Vector3dVector(vis_tgt[:, :3])
            vis_tgt_pcd.colors = o3d.utility.Vector3dVector(vis_tgt[:, 3:])
            vis_deform_tgt_pcd = o3d.geometry.PointCloud()
            vis_deform_tgt_pcd.points = o3d.utility.Vector3dVector(vis_deform_tgt[:, :3])
            vis_deform_tgt_pcd.colors = o3d.utility.Vector3dVector(vis_deform_tgt[:, 3:])

            vis_tgt_pcd = keypoints_to_spheres(vis_tgt_pcd, radius=0.1)
            vis_deform_tgt_pcd = keypoints_to_spheres(vis_deform_tgt_pcd, radius=0.1)

            vis_tgt_scene = [pcd_tgt] + [vis_tgt_pcd]
            vis_ref_scene = [pcd_ref] + [vis_deform_tgt_pcd]

            # Compute scale amounts for decent visualization
            fp_tgt_bounds = o3d_geometry_list_aabb([pcd_tgt])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
            fp_ref_bounds = o3d_geometry_list_aabb([pcd_ref])

            fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
            fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

            resize_tgt_rate = IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0] if fp_tgt_lengths[0] > fp_tgt_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2]
            resize_ref_rate = IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0] if fp_ref_lengths[0] > fp_ref_lengths[2] else IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2]
            vis_tgt_scene = o3d_geometry_list_scale(vis_tgt_scene, resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
            vis_ref_scene = o3d_geometry_list_scale(vis_ref_scene, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

            # Compute shift amounts from bounding box
            fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene[:len([pcd_tgt])])  # NOTE: We only use the original scene values instead of bounding boxes which can overhshoot
            fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene[:len([pcd_ref])])

            vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
            vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
            vis_tgt_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])
            vis_ref_displacement = np.array([IDEAL_VIS_LENGTH[0] + args.vis_margin, 0., 0.])

            vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
            vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

            # Fix both scenes' ground level
            tgt_ground = o3d_geometry_list_aabb(vis_tgt_scene[:len([pcd_tgt])])[1, 1]
            ref_ground = o3d_geometry_list_aabb(vis_ref_scene[:len([pcd_ref])])[1, 1]

            vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], -tgt_ground, vis_tgt_shift[2]])
            vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], -ref_ground, vis_ref_shift[2]])
            geometry_list = vis_tgt_scene + vis_ref_scene

            if args.vis_local_match_mode == 'region_2d_match_overlay':
                pbr_geometry_list = []

                for geo_idx, geom in enumerate(geometry_list):
                    if geo_idx in [len(vis_tgt_scene) - 1, len(vis_tgt_scene) + len(vis_ref_scene) - 1]:  # Make original query points translucent
                        material = o3d.visualization.rendering.MaterialRecord()
                        material.shader = 'defaultLitTransparency'
                        material.base_color = [0.4, 0.4, 0.4, 0.7]  # RGBA, adjust A for transparency
                    else:
                        if getattr(geom, 'textures', None) is not None:
                            material = o3d.visualization.rendering.MaterialRecord()
                            material.shader = 'defaultUnlit'
                            material.albedo_img = o3d.geometry.Image(geom.textures[0])
                        else:
                            material = None

                    pbr_geometry_list.append({'name': f'geom_{geo_idx}', 'geometry': geom, 'material': material})

                # NOTE: Currently this only supports screen display (TODO: Have this to support image saving)
                o3d.visualization.draw(pbr_geometry_list, show_skybox=False)
            else:
                o3d.visualization.draw_geometries(geometry_list)
