from threed_front.data_utils import KeypointSceneBuffer, keybuff_collate_fn, build_dense_3d_scene
import torch
import torch.nn as nn
from threed_front.feature_field import (
    KeyPointFeatureField,
    KeyPointSimpleField,
    KeyPointMeanField,
)
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from glob import glob
from threed_front.utils import (
    print_state,
    RollingSampler,
    InfoNCE,
    keypoints_to_spheres,
    o3d_geometry_list_shift,
    o3d_geometry_list_scale,
    o3d_geometry_list_aabb
)
import argparse
import numpy as np
import random
import wandb
from matplotlib import colormaps
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2


# Ideal scene bounding box size for visualization
IDEAL_VIS_LENGTH = np.array([12.0, 3.2, 12.0])


class Trainer():
    def __init__(self, cfg, log_dir, user_cfg_dict):
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.log_dir = log_dir
        self.user_cfg_dict = user_cfg_dict  # Dictionary of values in cfg that are specified by the user
        self.mode = cfg.mode
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.cfg.load_feature_field is not None:
            print(f"Loading model from {self.cfg.load_feature_field}")
            self.feature_field = torch.load(self.cfg.load_feature_field).to(self.device)
            self.feature_field.update_cfg(user_cfg_dict)
        else:
            if self.cfg.model_type == 'feature_field':
                self.feature_field = KeyPointFeatureField(cfg, device=self.device)
            elif self.cfg.model_type == 'simple_field':
                self.feature_field = KeyPointSimpleField(cfg, device=self.device)
            elif self.cfg.model_type == 'mean_field':
                self.feature_field = KeyPointMeanField(cfg, device=self.device)
            else:
                raise NotImplementedError("Other model types not implemented")

        if self.mode == 'test':
            self.feature_field = self.feature_field.eval()

        # Setup configs for training
        self.epochs = cfg.epochs
        self.optimizer = optim.Adam(self.feature_field.parameters(), lr=self.cfg.learning_rate)
        self.loss_type = cfg.loss_type
        self.triplet_margin = cfg.triplet_margin
        self.margin_loss = nn.TripletMarginLoss(self.triplet_margin)
        self.infonce_temp = cfg.infonce_temp

        # Optionally set scheduler
        if self.cfg.scheduler == 'steplr':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.epochs * 0.4), gamma=0.2)
        else:
            self.scheduler = None

        if self.loss_type == 'infonce_paired':
            self.infonce_loss = InfoNCE(self.infonce_temp, negative_mode='paired')
        else:
            self.infonce_loss = InfoNCE(self.infonce_temp, negative_mode='unpaired')

        # Setup configs for scene buffer
        self.scene_buffer_size = cfg.scene_buffer_size
        self.scene_root = cfg.scene_root
        self.query_point_type = cfg.query_point_type
        self.num_cache_query = cfg.num_cache_query
        self.num_sample_query = cfg.num_sample_query
        self.full_scene_path_list = sorted(glob(os.path.join(self.scene_root, '**/*.npz')))

        # NOTE 1: Here we list up pairs of paths (pos_path & pair_pos_path) which are kept identical for training
        # NOTE 2: pos_path is used for loading scene information for pos / neg samples and pair_pos_path is used for loading scene information for pair_pos samples
        self.train_scene_path_pair_list = [(path, path) for path in self.full_scene_path_list if 'train' in path]
        self.test_scene_path_pair_list = [(path, path) for path in self.full_scene_path_list if 'test' in path]
        self.train_scene_path_pair_sampler = RollingSampler(self.train_scene_path_pair_list)
        self.test_scene_path_pair_sampler = RollingSampler(self.test_scene_path_pair_list)

        self.train_scene_buffer = None
        self.eval_scene_buffer = None
        self.train_loader = None
        self.eval_loader = None

    def reset_scene_buffer(self, mode):
        if mode == 'train':
            scene_path_pair_list = self.train_scene_path_pair_sampler.sample(self.scene_buffer_size)
            self.train_scene_buffer = KeypointSceneBuffer(
                scene_path_pair_list=scene_path_pair_list,
                cfg=self.cfg,
                device=self.device
            )
            self.train_loader = DataLoader(
                self.train_scene_buffer,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=keybuff_collate_fn
            )
        else:  # Set eval buffer
            scene_path_pair_list = self.test_scene_path_pair_sampler.sample(self.scene_buffer_size)
            self.test_scene_buffer = KeypointSceneBuffer(
                scene_path_pair_list=scene_path_pair_list,
                cfg=self.cfg,
                device=self.device
            )
            self.test_loader = DataLoader(
                self.test_scene_buffer,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=keybuff_collate_fn
            )

    def run(self):
        if self.mode == 'train':
            print("Begin Train!")
            self.run_train()
        else:
            print("Begin Test!")
            self.run_test()

    def run_train(self):
        for idx in range(self.epochs):
            self.feature_field.to(self.device)
            print(f"Epoch {idx} Training")
            if idx % self.cfg.update_buffer_every == 0:
                self.reset_scene_buffer('train')
            self.feature_field = self.feature_field.train()
            self.train(idx)
            print(f"Epoch {idx} Evaluation")
            if idx % self.cfg.update_buffer_every == 0:
                self.reset_scene_buffer('eval')
            self.feature_field = self.feature_field.eval()
            self.eval(idx)
            if self.scheduler is not None:
                self.scheduler.step()

            torch.save(self.feature_field.to('cpu'), os.path.join(self.log_dir, 'model.pth'))  # Save latest model

    def run_test(self):
        for eval_idx in range(self.cfg.eval_reps):
            self.reset_scene_buffer('eval')
            self.eval(eval_idx)

    def train(self, epoch):
        for batch_idx, (scene_pcd, query_pcd) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            num_batch_scenes = len(scene_pcd['pos'])  # For each triplet, number of scenes is fixed

            # Extract features
            if self.cfg.semantics_emb_type == 'class_vector':
                pos_feats = self.feature_field(query_pcd['pos'].to(self.device), scene_pcd['pos'].to(self.device))  # (N_batch, N_query, D_feat)
                pair_pos_feats = self.feature_field(query_pcd['pair_pos'].to(self.device), scene_pcd['pair_pos'].to(self.device))  # (N_batch, N_query, D_feat)
                neg_feats = self.feature_field(query_pcd['neg'].to(self.device), scene_pcd['neg'].to(self.device))  # (N_batch, N_query, D_feat)
            else:
                raise NotImplementedError("Other embeddings not supported")

            emb_size = pos_feats.shape[2]
            num_query = pos_feats.shape[1]  # Fixed for all scene types
            pos_feats = pos_feats.reshape(num_batch_scenes * num_query, emb_size)
            pair_pos_feats = pair_pos_feats.reshape(num_batch_scenes * num_query, emb_size)
            neg_feats = neg_feats.reshape(num_batch_scenes * num_query, emb_size)

            if self.loss_type == "triplet_margin":
                loss = self.margin_loss(pos_feats, pair_pos_feats, neg_feats)
            elif self.loss_type == "infonce_unpaired":  # All negative samples are considered per each sample
                loss = self.infonce_loss(pos_feats, pair_pos_feats, neg_feats)
            elif self.loss_type == "infonce_paired":  # Only paired negative samples are considered per each sample
                loss = self.infonce_loss(pos_feats, pair_pos_feats, neg_feats[:, None, :])
            else:
                raise NotImplementedError("Other loss types not supported")

            print_dict = {
                'Iter': batch_idx,
                'Loss': loss.item(),
            }

            if self.cfg.wandb:
                wandb.log({"training_loss": loss.item()})

            if loss.requires_grad:  # Backprop if there are things to be optimized
                loss.backward()
                self.optimizer.step()

            print_state(print_dict)

    def eval(self, epoch):
        for batch_idx, (scene_pcd, query_pcd) in enumerate(self.test_loader):
            num_batch_scenes = len(scene_pcd['pos'])  # For each triplet, number of scenes is fixed

            # Extract features
            with torch.no_grad():
                if self.cfg.semantics_emb_type == 'class_vector':
                    pos_feats = self.feature_field(query_pcd['pos'].to(self.device), scene_pcd['pos'].to(self.device))  # (N_batch, N_query, D_feat)
                    pair_pos_feats = self.feature_field(query_pcd['pair_pos'].to(self.device), scene_pcd['pair_pos'].to(self.device))  # (N_batch, N_query, D_feat)
                    neg_feats = self.feature_field(query_pcd['neg'].to(self.device), scene_pcd['neg'].to(self.device))  # (N_batch, N_query, D_feat)
                else:
                    raise NotImplementedError("Other embeddings not supported")

                emb_size = pos_feats.shape[2]
                num_query = pos_feats.shape[1]  # Fixed for all scene types
                pos_feats = pos_feats.reshape(num_batch_scenes * num_query, emb_size)
                pair_pos_feats = pair_pos_feats.reshape(num_batch_scenes * num_query, emb_size)
                neg_feats = neg_feats.reshape(num_batch_scenes * num_query, emb_size)

                if self.loss_type == "triplet_margin":
                    loss = self.margin_loss(pos_feats, pair_pos_feats, neg_feats)
                elif self.loss_type == "infonce_unpaired":  # All negative samples are considered per each sample
                    loss = self.infonce_loss(pos_feats, pair_pos_feats, neg_feats)
                elif self.loss_type == "infonce_paired":  # Only paired negative samples are considered per each sample
                    loss = self.infonce_loss(pos_feats, pair_pos_feats, neg_feats[:, None, :])
                else:
                    raise NotImplementedError("Other loss types not supported")

            print_dict = {
                'Iter': batch_idx,
                'Loss': loss.item(),
            }

            if self.cfg.wandb:
                wandb.log({"validation_loss": loss.item()})

            print_state(print_dict)

            # Optionally visualize query point matches
            if args.vis_query_mode is not None:
                vis_pos_feats = pos_feats.reshape(num_batch_scenes, num_query, emb_size)
                vis_pair_pos_feats = pair_pos_feats.reshape(num_batch_scenes, num_query, emb_size)
                vis_neg_feats = neg_feats.reshape(num_batch_scenes, num_query, emb_size)

                # Features are offloaded to CPU to save GPU memory
                vis_pos_feats = vis_pos_feats.cpu()
                vis_pair_pos_feats = vis_pair_pos_feats.cpu()
                vis_neg_feats = vis_neg_feats.cpu()

                for scene_idx in range(num_batch_scenes):
                    num_buffer_batches = self.scene_buffer_size // self.cfg.batch_size + 1 if self.scene_buffer_size % self.cfg.batch_size != 0 else self.scene_buffer_size // self.cfg.batch_size
                    vis_sample_idx = num_buffer_batches * num_batch_scenes * epoch + batch_idx * num_batch_scenes + scene_idx

                    # Build original dense 3D scene for visualization
                    dense_scenes = {}
                    dense_scenes['pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pos'][scene_idx], 'pos')
                    dense_scenes['pair_pos'] = build_dense_3d_scene(scene_pcd['scene_path']['pair_pos'][scene_idx], 'pair_pos')
                    dense_scenes['neg'] = build_dense_3d_scene(scene_pcd['scene_path']['neg'][scene_idx], 'neg')

                    dist_self = (vis_pos_feats[scene_idx][:, None, :] - vis_pos_feats[scene_idx][None, :, :]).norm(dim=-1)  # (N_query, N_query)
                    dist_self[range(num_query), range(num_query)] = np.inf
                    dist_pos2pair_pos = (vis_pos_feats[scene_idx][:, None, :] - vis_pair_pos_feats[scene_idx][None, :, :]).norm(dim=-1)  # (N_query, N_query)
                    dist_pos2neg = (vis_pos_feats[scene_idx][:, None, :] - vis_neg_feats[scene_idx][None, :, :]).norm(dim=-1)  # (N_query, N_query)

                    vis_query_pos = query_pcd['pos'][scene_idx].points_packed().cpu().numpy()
                    vis_query_pair_pos = query_pcd['pair_pos'][scene_idx].points_packed().cpu().numpy()
                    vis_query_neg = query_pcd['neg'][scene_idx].points_packed().cpu().numpy()

                    idx_color = np.linspace(0., 1., num_query)  # (N_query, )
                    idx_color = colormaps['jet'](idx_color, alpha=False, bytes=False)[:, :3]

                    sort_idx = vis_query_pos[:, 0].argsort().argsort()
                    idx_color = idx_color[sort_idx]

                    if args.vis_query_mode in ['raw_pos', 'raw_pair_pos', 'raw_neg']:
                        if args.vis_query_mode == 'raw_pair_pos':
                            vis_ref_dense = dense_scenes['pair_pos']
                        elif args.vis_query_mode == 'raw_pos':
                            vis_ref_dense = dense_scenes['pos']
                        else:
                            vis_ref_dense = dense_scenes['neg']

                        # Compute scale amounts for decent visualization
                        fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])
                        fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_dense)

                        fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                        fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                        resize_tgt_rate = min(IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0], IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2])
                        resize_ref_rate = min(IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0], IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2])
                        vis_tgt_scene = o3d_geometry_list_scale(dense_scenes['pos'], resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                        vis_ref_scene = o3d_geometry_list_scale(vis_ref_dense, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                        # Compute shift amounts from bounding box
                        fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene)
                        fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene)

                        vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                        vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                        vis_tgt_displacement = np.array([(fp_tgt_bounds[0, 0] - fp_tgt_bounds[1, 0]) + args.vis_margin, 0., 0.])
                        vis_ref_displacement = np.array([(fp_ref_bounds[0, 0] - fp_ref_bounds[1, 0]) + args.vis_margin, 0., 0.])

                        vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                        vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                        vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], 0., vis_tgt_shift[2]])
                        vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], 0., vis_ref_shift[2]])
                        geometry_list = vis_tgt_scene + vis_ref_scene
                        self.visualize_geometry(geometry_list, vis_sample_idx, scene_idx, pcd_resize_rate=(resize_ref_rate + resize_tgt_rate) / 2.)

                    elif args.vis_query_mode in ['feat_dist', 'feat_dist_intra', 'feat_dist_neg']:
                        # Choose from randomly sampling query points for visualization or showing them all
                        random_sample = (self.cfg.feat_dist_sample_repeats != -1)

                        if random_sample:
                            num_dist_samples = self.cfg.feat_dist_sample_repeats
                        else:
                            num_dist_samples = vis_query_pos.shape[0]

                        for dist_sample_idx in range(0, num_dist_samples, args.feat_dist_query_subsample):
                            if random_sample:
                                vis_idx = random.randint(0, vis_query_pos.shape[0] - 1)
                            else:
                                vis_idx = dist_sample_idx

                            if args.vis_query_mode == 'feat_dist':
                                vis_feat_dist = dist_pos2pair_pos[vis_idx].cpu().numpy()
                                vis_ref_query_coords = vis_query_pair_pos
                                vis_ref_dense = dense_scenes['pair_pos']
                            elif args.vis_query_mode == 'feat_dist_intra':
                                vis_feat_dist = dist_self[vis_idx].cpu().numpy()
                                vis_feat_dist[vis_idx] = 0.0  # Reset infinite distance to zero
                                vis_ref_query_coords = vis_query_pair_pos
                                vis_ref_dense = dense_scenes['pos']
                            elif args.vis_query_mode == 'feat_dist_neg':
                                vis_feat_dist = dist_pos2neg[vis_idx].cpu().numpy()
                                vis_ref_query_coords = vis_query_neg
                                vis_ref_dense = dense_scenes['neg']
                            else:
                                raise NotImplementedError("Other query modes not supported")

                            vis_pos_pcd = o3d.geometry.PointCloud()
                            vis_pos_pcd.points = o3d.utility.Vector3dVector(vis_query_pos[vis_idx:vis_idx+1, :3])
                            vis_pos_pcd.colors = o3d.utility.Vector3dVector(np.array([[1., 0., 0.]]))

                            if self.cfg.vis_truncate_rate > 0.:
                                cutoff_dist = np.sort(vis_feat_dist)[-int(len(vis_feat_dist) * self.cfg.vis_truncate_rate)]
                                valid_mask = vis_feat_dist < cutoff_dist
                                valid_feat_dist = vis_feat_dist[valid_mask]
                                valid_feat_dist = (valid_feat_dist - valid_feat_dist.min()) / (valid_feat_dist.max() - valid_feat_dist.min())
                                valid_feat_dist = valid_feat_dist ** self.cfg.vis_gamma
                                valid_feat_color = colormaps['jet'](valid_feat_dist, alpha=False, bytes=False)[:, :3]
                                vis_ref = np.concatenate([vis_ref_query_coords[valid_mask], valid_feat_color], axis=-1)  # (N_query, 6)
                            else:
                                vis_feat_dist = (vis_feat_dist - vis_feat_dist.min()) / (vis_feat_dist.max() - vis_feat_dist.min())
                                vis_feat_dist = vis_feat_dist ** self.cfg.vis_gamma
                                vis_feat_color = colormaps['jet'](vis_feat_dist, alpha=False, bytes=False)[:, :3]
                                vis_ref = np.concatenate([vis_ref_query_coords, vis_feat_color], axis=-1)  # (N_query, 6)

                            # Compute scale amounts for decent visualization
                            fp_tgt_bounds = o3d_geometry_list_aabb(dense_scenes['pos'])
                            fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_dense)

                            fp_tgt_lengths = fp_tgt_bounds[0] - fp_tgt_bounds[1]
                            fp_ref_lengths = fp_ref_bounds[0] - fp_ref_bounds[1]

                            resize_tgt_rate = min(IDEAL_VIS_LENGTH[0] / fp_tgt_lengths[0], IDEAL_VIS_LENGTH[2] / fp_tgt_lengths[2])
                            resize_ref_rate = min(IDEAL_VIS_LENGTH[0] / fp_ref_lengths[0], IDEAL_VIS_LENGTH[2] / fp_ref_lengths[2])
                            vis_tgt_scene = o3d_geometry_list_scale(dense_scenes['pos'], resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                            vis_ref_scene = o3d_geometry_list_scale(vis_ref_dense, resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                            vis_ref_pcd = o3d.geometry.PointCloud()
                            vis_ref_pcd.points = o3d.utility.Vector3dVector(vis_ref[:, :3])
                            vis_ref_pcd.colors = o3d.utility.Vector3dVector(vis_ref[:, 3:])

                            vis_tgt_pcd = o3d_geometry_list_scale([keypoints_to_spheres(vis_pos_pcd, radius=0.3)], resize_tgt_rate, fp_tgt_bounds.mean(0).reshape(3, 1))
                            vis_ref_pcd = o3d_geometry_list_scale([keypoints_to_spheres(vis_ref_pcd, radius=0.1)], resize_ref_rate, fp_ref_bounds.mean(0).reshape(3, 1))

                            # Compute shift amounts from bounding box
                            fp_tgt_bounds = o3d_geometry_list_aabb(vis_tgt_scene)
                            fp_ref_bounds = o3d_geometry_list_aabb(vis_ref_scene)

                            vis_tgt_centroid = (fp_tgt_bounds[0] + fp_tgt_bounds[1]) / 2.
                            vis_ref_centroid = (fp_ref_bounds[0] + fp_ref_bounds[1]) / 2.
                            vis_tgt_displacement = np.array([(fp_tgt_bounds[0, 0] - fp_tgt_bounds[1, 0]) + args.vis_margin, 0., 0.])
                            vis_ref_displacement = np.array([(fp_ref_bounds[0, 0] - fp_ref_bounds[1, 0]) + args.vis_margin, 0., 0.])

                            vis_tgt_shift = -(vis_tgt_centroid + vis_tgt_displacement / 2.)
                            vis_ref_shift = -(vis_ref_centroid - vis_ref_displacement / 2.)

                            vis_tgt_scene = o3d_geometry_list_shift(vis_tgt_scene, [vis_tgt_shift[0], 0., vis_tgt_shift[2]])
                            vis_ref_scene = o3d_geometry_list_shift(vis_ref_scene, [vis_ref_shift[0], 0., vis_ref_shift[2]])

                            vis_tgt_pcd = o3d_geometry_list_shift(vis_tgt_pcd, [vis_tgt_shift[0], 0., vis_tgt_shift[2]])
                            vis_ref_pcd = o3d_geometry_list_shift(vis_ref_pcd, [vis_ref_shift[0], 0., vis_ref_shift[2]])

                            geometry_list = vis_tgt_scene + vis_ref_scene + vis_tgt_pcd + vis_ref_pcd

                            self.visualize_geometry(
                                geometry_list,
                                vis_sample_idx,
                                scene_idx,
                                repeat_idx=dist_sample_idx,
                                num_repeats=num_dist_samples,
                                close_window_at_end=(scene_idx == num_batch_scenes - 1) and (dist_sample_idx == num_dist_samples - 1),
                                pcd_resize_rate=(resize_ref_rate + resize_tgt_rate) / 2.
                            )
                    else:
                        raise NotImplementedError("Other visualization types not supported")

    def visualize_geometry(self, geometry_list, vis_sample_idx=0, scene_idx=0, repeat_idx=0, num_repeats=1, close_window_at_end=False, pcd_resize_rate=1.):  # Helper function for visualizing each scene in batch
        # vis_sample_idx is the visualization index to use for saving rendered image / video
        # scene_idx is index of scene within batch, repeat_idx is the index of the repeat in repeated visualization (e.g., feat_dist visualization)
        # NOTE: Indices only matter for render_img and render_video modes

        if self.cfg.vis_method == "screen":
            o3d.visualization.draw_geometries(geometry_list)
        elif self.cfg.vis_method in ["render_img", "render_video", "render_img_top_down"]:
            if not os.path.exists(os.path.join(self.cfg.log_dir, f'{self.cfg.vis_query_mode}_{self.cfg.query_point_type}_rendering')):
                os.makedirs(os.path.join(self.cfg.log_dir, f'{self.cfg.vis_query_mode}_{self.cfg.query_point_type}_rendering'), exist_ok=True)

            # Initialize open3d visualizer for new scene
            if repeat_idx == 0:  # Generate window only for first frame
                self.visualizer = o3d.visualization.Visualizer()
                self.visualizer.create_window()
                self.visualizer.get_render_option().point_size = self.cfg.vis_point_size * pcd_resize_rate

            for geometry in geometry_list:
                self.visualizer.add_geometry(geometry)
                self.visualizer.update_geometry(geometry)

            # Change to top-down view
            ctr = self.visualizer.get_view_control()
            rot = np.eye(4)
            rot[:3, :3] = R.from_euler('x', 80, degrees=True).as_matrix()
            cam = ctr.convert_to_pinhole_camera_parameters()
            cam.extrinsic = cam.extrinsic @ rot

            # Fix camera parameters to near-orthographic
            new_cam_extrinsic = np.copy(cam.extrinsic)
            new_cam_extrinsic[0, -1] = o3d_geometry_list_aabb(geometry_list).mean(0)[0]  # Make camera centered around scene
            new_cam_extrinsic[2, -1] += 300.
            cam.extrinsic = new_cam_extrinsic
            new_cam_intrinsic = np.copy(cam.intrinsic.intrinsic_matrix)
            new_cam_intrinsic[0, 0] *= 20.
            new_cam_intrinsic[1, 1] *= 20.
            cam.intrinsic.intrinsic_matrix = new_cam_intrinsic
            ctr.convert_from_pinhole_camera_parameters(cam, True)

            self.visualizer.poll_events()
            self.visualizer.update_renderer()

            if self.cfg.vis_method == "render_img":
                self.visualizer.capture_screen_image(os.path.join(self.cfg.log_dir, f'{self.cfg.vis_query_mode}_{self.cfg.query_point_type}_rendering', f'render_{vis_sample_idx}_repeat_{repeat_idx}.png'))
            elif self.cfg.vis_method == "render_img_top_down":
                self.visualizer.capture_screen_image(os.path.join(self.cfg.log_dir, f'{self.cfg.vis_query_mode}_{self.cfg.query_point_type}_rendering', f'render_top_{vis_sample_idx}_repeat_{repeat_idx}.png'))
                ctr = self.visualizer.get_view_control()
                rot = np.eye(4)
                rot[:3, :3] = R.from_euler('x', -180, degrees=True).as_matrix()
                cam = ctr.convert_to_pinhole_camera_parameters()
                cam.extrinsic = cam.extrinsic @ rot
                ctr.convert_from_pinhole_camera_parameters(cam, True)
                ctr.set_zoom(0.4)

                self.visualizer.poll_events()
                self.visualizer.update_renderer()
                self.visualizer.capture_screen_image(os.path.join(self.cfg.log_dir, f'{self.cfg.vis_query_mode}_{self.cfg.query_point_type}_rendering', f'render_down_{vis_sample_idx}_repeat_{repeat_idx}.png'))
            else:  # Render to video
                frame = self.visualizer.capture_screen_float_buffer()
                frame = (255 * np.asarray(frame)).astype(np.uint8)

                if repeat_idx == 0:
                    self.video = cv2.VideoWriter(
                        os.path.join(self.cfg.log_dir, f'{self.cfg.vis_query_mode}_{self.cfg.query_point_type}_rendering', f'render_{vis_sample_idx}' + '.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        5.,
                        (frame.shape[1], frame.shape[0]))

                self.video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if repeat_idx == num_repeats - 1:
                    self.video.release()

            self.visualizer.clear_geometries()

            repeat_idx += 1
            if close_window_at_end:
                self.visualizer.destroy_window()
        else:
            raise NotImplementedError("Other visualization methods not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General configs
    parser.add_argument("--log_dir", help="Log directory for saving experiment results", default="./log/")
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)
    parser.add_argument("--run_name", help="Optional nametag for run logging in wandb", default=None, type=str)
    parser.add_argument("--wandb", help="Optionally log metrics to wandb", action='store_true')
    parser.add_argument("--eval_reps", help="Number of evaluation repititions to make", default=1, type=int)

    # Dataset configs
    parser.add_argument("--point_feat_extractor", help="Type of point feature extractor to use for dataset generation", default=None, type=str)
    parser.add_argument("--feat_sample_points", type=int, help="Number of points to sample per object model for feature extraction", default=2048)
    parser.add_argument("--scene_sample_points", type=int, help="Number of points to sample per object model for scene generation", default=50)
    parser.add_argument("--obj_json", help=".json file containing information on 3D-FUTURE meshes", default="./data/3D-FUTURE-model/model_info.json")
    parser.add_argument("--feat_pca_root", help="Root folder containing PCA components for point feature dimension reduction (currently used for Vector Neurons)", default="./data/3d_future_pca/")
    parser.add_argument("--point_sample_root", help="Root folder containing point samples from object mesh files (currently used for scene generation without feature extraction)", default="./data/3d_future_point_samples/")
    parser.add_argument("--random_group_topk_range", type=int, help="Minimum and maximum value for top-K nearest object sampling during random group query sampling", default=[1, 4], nargs=2)

    # Training configs
    parser.add_argument("--model_type", help="Type of model to train or test", default="feature_field", type=str)
    parser.add_argument("--scene_root", help="Root directory containing scene data", required=True, type=str)
    parser.add_argument("--batch_size", help="Batch size to use for training", default=4, type=int)
    parser.add_argument("--epochs", help="Number of epochs to run for training", default=10000, type=int)
    parser.add_argument("--mode", help="Mode to use training (either train or test)", default="train", type=str)
    parser.add_argument("--learning_rate", help="Learning rate for training feature field", default=0.0001, type=float)
    parser.add_argument("--scheduler", help="Learning rate scheduling for training feature field", default='steplr', type=str)
    parser.add_argument("--loss_type", help="Type of loss to use for training", default="infonce_paired", type=str)
    parser.add_argument("--triplet_margin", help="Margin value for triple loss", default=0.5, type=float)
    parser.add_argument("--infonce_temp", help="InfoNCE temperature parameter", default=0.2, type=float)
    parser.add_argument("--scene_buffer_size", help="Size of buffer to use for loading scenes each epoch", default=32, type=int)
    parser.add_argument("--query_point_type", help="Type of query point sampling to use for generating trianing samples", default="obj_points", type=str)
    parser.add_argument("--num_vert_split", help="Number of vertical splits to make for floorplan queries", default=1, type=int)
    parser.add_argument("--force_up_margin", help="Forced upper margin for floorplan queries (defaults to using maximum height of walls)", default=0.5, type=float)
    parser.add_argument("--force_low_margin", help="Forced lower margin for floorplan queries (defaults to using minimum height of walls)", default=0.5, type=float)
    parser.add_argument("--num_cache_query", help="Number of query point locations to cacher per scene", default=50000, type=int)
    parser.add_argument("--num_sample_query", help="Number of query points to sample per scene", default=32, type=int)
    parser.add_argument("--update_buffer_every", help="Number of epochs before updating buffer", default=32, type=int)
    parser.add_argument("--num_classes", help="Number of semantics classes in objects", default=8, type=int)
    parser.add_argument("--fp_feat_type", help="Type of floorplan features to use", default="learned", type=str)
    parser.add_argument("--obj_point_query_scale_factor", help="Scale factor to use for object point query sampling", default=None, type=float, nargs="+")
    parser.add_argument("--fp_point_type", help="Type of floorplan points to use", default="wireframe", type=str)
    parser.add_argument("--fp_sample_step_size", help="Step size (contour, height) for generating floorplan sampled points", default=[0.3, 0.3], type=float, nargs=2)
    parser.add_argument("--fp_label_type", help="Type of floorplan labeling to use", default="single", type=str)
    parser.add_argument("--query_obj_match_mode", help="Type of solver to use for obtaining object match points during query sampling when query_point_type is obj_match_points", default="jv", type=str)
    parser.add_argument("--query_obj_match_add_bbox", help="Optionally add bounding box proximal points during query sampling when query_point_type is obj_match_points", action="store_true")

    # Network configs
    parser.add_argument("--load_feature_field", help="Optionally load pre-trained feature field", default=None, type=str)
    parser.add_argument("--agg_method", help="Aggregation method for gathering neighboring points", default="ball_query", type=str)
    parser.add_argument("--agg_k", help="Number of nearest neighbors to extract", default=30, type=int)
    parser.add_argument("--point_descriptor_size", help="Size of point descriptor to use for transformer inputs", default=32, type=int)
    parser.add_argument("--agg_radius", help="Radius to use for ball query aggregation", default=0.5, type=float)
    parser.add_argument("--agg_radius_limit", help="Maximum number of points to be aggregated for ball query aggregation", default=100, type=int)
    parser.add_argument("--semantics_emb_size", help="Size of semantics embedding", default=32, type=int)
    parser.add_argument("--semantics_emb_type", help="Type of semantics embedding", default="class_vector", type=str)
    parser.add_argument("--gr_feat_type", help="Type of optional ground embedding to use", default=None, type=str)
    parser.add_argument("--height_emb_size", help="Size of height embedding", default=0, type=int)
    parser.add_argument("--distance_emb_size", help="Size of distance embedding", default=32, type=int)
    parser.add_argument("--distance_emb_type", help="Type of distance embedding to use for training", default="direct_learned", type=str)
    parser.add_argument("--train_semantics_emb", help="Optionally train semantics embedding (default is to freeze)", action="store_true")
    parser.add_argument("--out_dim", help="Size of output embedding from feature field", default=256)
    parser.add_argument("--normalize_feats", help="If True, normalize network outputs to unit norm", action="store_true")
    parser.add_argument("--agg_exp_temp", help="Exponential temperature to use for aggregation in simple feature field", default=1.0, type=float)
    parser.add_argument("--transformer_num_heads", help="Number of heads for Transformer", default=8, type=int)
    parser.add_argument("--transformer_num_layers", help="Number of layers for Transformer", default=6, type=int)

    # Evaluation configs
    parser.add_argument("--vis_query_mode", help="Mode for visualizing query point matches during evaluation", default=None, type=str)
    parser.add_argument("--vis_margin", help="Amount of margins to apply for reference scene during visualization", default=5., type=float)
    parser.add_argument("--feat_dist_sample_repeats", help="Number of repeats to make per scene or feature distance visualization", default=1, type=int)
    parser.add_argument("--feat_dist_query_subsample", help="Query point subsampling rate for faster feature distance visualization", default=1, type=int)
    parser.add_argument("--vis_method", help="Type of visualization to perform", default="screen", type=str)
    parser.add_argument("--vis_gamma", help="Gamma value to use for visualization", default=1., type=float)
    parser.add_argument("--vis_truncate_rate", help="Rate of feature distances to truncate during visualization", default=0.0, type=float)
    parser.add_argument("--vis_point_size", help="Size of point for rendering views", default=2., type=float)

    # TODO: Set for temporary visualization (will be removed at a later time)
    parser.add_argument("--vis_agg_radius", help="Radius for visualizing aggregated field values", default=0., type=float)
    parser.add_argument("--vis_agg_k", help="Upper bound for ball query when visualizing aggregated field values", default=500, type=int)
    parser.add_argument("--vis_agg_weight_method", help="Weighting scheme when visualizing aggregated field values", default='equal', type=str)
    args = parser.parse_args()

    # Keep set of user-specified non-default argument values
    mod_args_dict = {}  # Used for modifying pre-trained models' configs
    args_dict = vars(args)
    for k, v in args_dict.items():
        if v != parser.get_default(k):
            mod_args_dict[k] = v

    # Fix seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Optionally initialize wandb for logging
    if args.wandb:
        wandb.init(project="neu-rel-field", config=vars(args), name=args.run_name)

    trainer = Trainer(args, log_dir=args.log_dir, user_cfg_dict=mod_args_dict)
    trainer.run()
