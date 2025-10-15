import torch.nn as nn
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points, knn_gather, ball_query
from pytorch3d.ops.utils import masked_gather
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import namedtuple


def adaptive_k_from_radius(r: float):
    # Return number of aggregation points from designated radius
    key_points = np.array([0.15, 0.3, 0.5, 0.75, 1., 1.5, 2., 3.])
    key_num_agg = np.array([50., 50., 105., 105., 160., 343., 473., 557.])  # Values found from manual inspection
    adaptive_k = int(np.interp(np.array([r]), key_points, key_num_agg).item())
    return adaptive_k


# Positional encodings
class PositionalEncoding(torch.nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(PositionalEncoding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs - 1), N_freqs)

    def forward(self, x, omit_orig=True):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        if omit_orig:
            out = []
        else:
            out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class DistanceEmb(nn.Module):
    def __init__(self, dist_dim):
        super(DistanceEmb, self).__init__()
        sin_dim = dist_dim // 2
        self.pos_emb = PositionalEncoding(1, sin_dim)
        self.dist_mlp = nn.Sequential(
            nn.Linear(sin_dim * 2, dist_dim),
            nn.GELU(),
            nn.Linear(dist_dim, dist_dim)
        )

    def forward(self, x):  # x is of shape (N_batch, 1)
        x = self.pos_emb(x)
        x = self.dist_mlp(x)
        return x


class DirectDistanceEmb(nn.Module):
    def __init__(self, dist_dim):
        super(DirectDistanceEmb, self).__init__()
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, dist_dim // 2),
            nn.ReLU(),
            nn.Linear(dist_dim // 2, dist_dim)
        )

    def forward(self, x):  # x is of shape (N_batch, 1)
        x = self.dist_mlp(x)
        return x


class AbstractFeatureField(nn.Module):
    def __init__(self, cfg, device: torch.device):
        super(AbstractFeatureField, self).__init__()
        self.cfg = cfg
        self.device = device

    def update_cfg(self, update_dict):
        cfg_dict = {k: v for k, v in vars(self.cfg).items()}
        Config = namedtuple('Config', tuple(set(tuple(cfg_dict) + tuple(update_dict.keys()))))
        cfg_dict.update(update_dict)
        self.cfg = Config(**cfg_dict)


class KeyPointFeatureField(AbstractFeatureField):
    def __init__(self, cfg, device: torch.device):
        super(KeyPointFeatureField, self).__init__(cfg, device)

        # Set aggregation configs
        self.agg_method = getattr(cfg, "agg_method", "ball_query")
        self.agg_k = getattr(cfg, "agg_k", 30)
        self.agg_radius = getattr(cfg, "agg_radius", 0.5)
        self.agg_radius_limit = getattr(cfg, "agg_radius_limit", 30)

        # Set semantics embedding
        self.semantics_emb_size = getattr(cfg, 'semantics_emb_size', 32)

        if self.semantics_emb_size != 0:
            if getattr(self.cfg, "fp_label_type", "single") == "single":
                # Include additional label for floorplan corners (treated as label 0)
                self.semantics_emb = nn.Embedding(cfg.num_classes + 1, self.semantics_emb_size, device=self.device)
            else:
                # Include additional label for ceiling, ground, and wall (treated as label 0, 1, 2, respectively)
                self.semantics_emb = nn.Embedding(cfg.num_classes + 3, self.semantics_emb_size, device=self.device)

            if not getattr(cfg, 'train_semantics_emb', False):  # Optionally freeze semantics embedding
                for param in self.semantics_emb.parameters():
                    param.requires_grad = False
        else:
            self.semantics_emb = None

        # Set distance encoding
        self.distance_emb_size = getattr(cfg, 'distance_emb_size', 32)
        if self.cfg.distance_emb_type == 'fixed':
            self.dist_emb = PositionalEncoding(1, self.distance_emb_size // 2)
        elif self.cfg.distance_emb_type == 'direct_learned':
            self.dist_emb = DirectDistanceEmb(self.distance_emb_size).to(self.device)
        else:  # Learned distance embeddings
            self.dist_emb = DistanceEmb(self.distance_emb_size).to(self.device)

        # Initiate Transformer
        encoder_dim = self.semantics_emb_size + self.distance_emb_size
        encoder_layer = nn.TransformerEncoderLayer(encoder_dim, nhead=self.cfg.transformer_num_heads, device=self.device, batch_first=True)
        encoder_norm = nn.LayerNorm(encoder_dim, elementwise_affine=True, device=self.device)
        self.feat_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.cfg.transformer_num_layers,
            norm=encoder_norm,
        ).to(self.device)

        # Set <SUMMARY> token embedding, concatenated at the beginning of each sequence
        self.summary_emb = nn.Embedding(1, encoder_dim).to(self.device)

        # Set final linear layer
        self.final_lin = nn.Linear(encoder_dim, cfg.out_dim).to(self.device)
        self.normalize_feats = getattr(cfg, 'normalize_feats', False)

    def forward(self, query_pcd: Pointclouds, scene_pcd: Pointclouds):
        # Extract neighborhood points (NOTE: features are in order [instance semantics descriptor])
        # NOTE: Distances are by default squared in ball_query, so this needs to be reverted
        if self.agg_method == "ball_query":
            max_k = adaptive_k_from_radius(self.agg_radius)
            max_k = min(scene_pcd.num_points_per_cloud().max().item(), max_k)  # max_k does not need to exceed maximum number of points
            agg_results = ball_query(
                query_pcd.points_padded(),
                scene_pcd.points_padded(),
                lengths1=query_pcd.num_points_per_cloud(),
                lengths2=scene_pcd.num_points_per_cloud(),
                K=max_k,
                radius=self.agg_radius,
                return_nn=True
            )
            agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
            agg_idx = agg_results.idx  # (N_batch, N_query, K)

            # Ensure maximum number of points to be aggregated via random sampling
            if max_k > self.agg_radius_limit:
                new_agg_dists = []
                new_agg_idx = []
                for scene_idx in range(len(scene_pcd)):  # Ensure sampling at non-zero distances
                    sampling_weight = (agg_dists[scene_idx] != 0.).float()
                    sampling_weight[sampling_weight.sum(dim=-1) == 0.] += 1.  # Ensure all weights sum to non-zero
                    sampling_idx = torch.multinomial(sampling_weight, self.agg_radius_limit, replacement=False)
                    new_agg_dists.append(torch.take_along_dim(agg_dists[scene_idx], sampling_idx, dim=-1))
                    new_agg_idx.append(torch.take_along_dim(agg_idx[scene_idx], sampling_idx, dim=-1))
                agg_dists = torch.stack(new_agg_dists)
                agg_idx = torch.stack(new_agg_idx)

            agg_feats = masked_gather(scene_pcd.features_padded(), agg_idx)  # (N_batch, N_query, K, 1 + 1 + D_desc)

        elif self.agg_method == "knn":
            agg_results = knn_points(
                query_pcd.points_padded(),
                scene_pcd.points_padded(),
                lengths1=query_pcd.num_points_per_cloud(),
                lengths2=scene_pcd.num_points_per_cloud(),
                K=self.agg_k
            )
            agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
            agg_idx = agg_results.idx  # (N_batch, N_query, K)
            agg_feats = knn_gather(
                scene_pcd.features_padded(),
                agg_idx,
                scene_pcd.num_points_per_cloud()
            )  # (N_batch, N_query, K, 1 + 1 + D_desc)
        else:
            raise NotImplementedError("Other aggregation types not supported")

        # Extract distance embeddings
        N_batch, N_query, K = agg_dists.shape
        agg_dists_flatten = agg_dists.reshape(-1, 1) / self.agg_radius  # NOTE: Normalized to ensure range in [0, 1]
        dists_emb = self.dist_emb(agg_dists_flatten)  # (N_batch * N_query * K, D_dist_emb)
        dists_emb = dists_emb.reshape(N_batch, N_query, K, self.distance_emb_size)  # (N_batch, N_query, K, D_dist_emb)

        # Extract semantics embeddings (additionally account for floorplan labels)
        if self.semantics_emb_size != 0:
            if getattr(self.cfg, "fp_label_type", "single") == "single":  # Single labels for wall or corners
                semantics_emb = self.semantics_emb(agg_feats[..., 1].long() + 1)  # (N_batch, N_query, K, D_sem_emb)
            else:  # Three additional labels for wall, ground, and floor
                semantics_emb = self.semantics_emb(agg_feats[..., 1].long() + 3)  # (N_batch, N_query, K, D_sem_emb)
        else:
            semantics_emb = torch.zeros(N_batch, N_query, K, 0, device=self.device)

        input_emb = torch.cat([dists_emb, semantics_emb], dim=-1)  # (N_batch, N_query, K, D_desc + D_dist_emb + D_sem_emb)

        # Flatten batch and query dimension for inference
        _, _, _, D_input = input_emb.shape
        input_emb_flatten = input_emb.reshape(-1, K, D_input)

        # Extract <SUMMARY> embedding
        summary_emb = self.summary_emb(torch.zeros([N_batch, N_query, 1, 1], dtype=torch.long, device=self.device))  # (N_batch, N_query, 1, D_input)
        summary_emb_flatten = summary_emb.reshape(-1, 1, D_input)

        # Make final input embedding and mask 
        input_emb_flatten = torch.cat([summary_emb_flatten, input_emb_flatten], dim=1)  # (N_batch * N_query, 1 + K, D_input)
        invalid_entry_mask = (agg_idx == -1).reshape(-1, K)  # (N_batch * N_query, K)
        input_mask_flatten = torch.cat([
            torch.zeros_like(invalid_entry_mask[:, :1]),
            invalid_entry_mask
        ], dim=-1)  # (N_batch * N_query, 1 + K)

        out_feats = self.feat_transformer(input_emb_flatten, src_key_padding_mask=input_mask_flatten)  # (N_batch * N_query, 1 + K, D_input)
        summary_feats = out_feats[:, 0, :]  # (N_batch * N_query, D_input)
        summary_feats = summary_feats.reshape(N_batch, N_query, -1)  # (N_batch, N_query, D_input)
        summary_feats = self.final_lin(summary_feats)

        # Optionally normalize features
        if self.normalize_feats:
            summary_feats = summary_feats / summary_feats.norm(dim=-1, keepdim=True)

        return summary_feats

    def update_cfg(self, update_dict):
        super(KeyPointFeatureField, self).update_cfg(update_dict)

        # Reset aggregation configs
        self.agg_method = getattr(self.cfg, "agg_method", "ball_query")
        self.agg_k = getattr(self.cfg, "agg_k", 30)
        self.agg_radius = getattr(self.cfg, "agg_radius", 0.5)
        self.agg_radius_limit = getattr(self.cfg, "agg_radius_limit", 30)


class KeyPointSimpleField(AbstractFeatureField):
    def __init__(self, cfg, device: torch.device):
        super(KeyPointSimpleField, self).__init__(cfg, device)

        # Set aggregation configs
        self.agg_method = getattr(cfg, "agg_method", "ball_query")
        self.agg_k = getattr(cfg, "agg_k", 30)
        self.agg_radius = getattr(cfg, "agg_radius", 0.5)
        self.agg_exp_temp = getattr(cfg, "agg_exp_temp", 1.0)  # Exponential temperature for distance-based aggregation

        # Set point descriptor size
        self.point_descriptor_size = getattr(self.cfg, 'point_descriptor_size', 48)

        # Set semantics embedding
        self.semantics_emb_size = getattr(cfg, 'semantics_emb_size', 32)

        if self.semantics_emb_size != 0:
            # Include additional label for floorplan corners (treated as label 0)
            self.semantics_emb = nn.Embedding(cfg.num_classes + 1, self.semantics_emb_size, device=self.device)

            # Freeze semantics embedding
            for param in self.semantics_emb.parameters():
                param.requires_grad = False
        else:
            self.semantics_emb = None

        # Optionally initialize learnable point descriptor for floorplan corners
        if self.cfg.fp_feat_type == 'learned':
            self.fp_point_desc = torch.nn.Parameter(torch.randn(self.point_descriptor_size, device=self.device), requires_grad=False)
        else:
            self.fp_point_desc = None

        # Set distance encoding
        self.distance_emb_size = getattr(cfg, 'distance_emb_size', 32)
        if self.cfg.distance_emb_type == 'fixed':
            self.dist_emb = PositionalEncoding(1, self.distance_emb_size // 2)
        elif self.cfg.distance_emb_type == 'direct_learned':
            self.dist_emb = DirectDistanceEmb(self.distance_emb_size).to(self.device)

            # Freeze distance embedding
            for param in self.dist_emb.parameters():
                param.requires_grad = False
        else:  # Learned distance embeddings
            self.dist_emb = DistanceEmb(self.distance_emb_size).to(self.device)

            # Freeze distance embedding
            for param in self.dist_emb.parameters():
                param.requires_grad = False

    def forward(self, query_pcd: Pointclouds, scene_pcd: Pointclouds):
        # Extract neighborhood points (NOTE: features are in order [instance semantics descriptor])
        if self.agg_method == "ball_query":
            agg_results = ball_query(
                query_pcd.points_padded(),
                scene_pcd.points_padded(),
                lengths1=query_pcd.num_points_per_cloud(),
                lengths2=scene_pcd.num_points_per_cloud(),
                K=adaptive_k_from_radius(self.agg_radius),
                radius=self.agg_radius,
                return_nn=True
            )
            agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
            agg_idx = agg_results.idx  # (N_batch, N_query, K)
            agg_feats = masked_gather(scene_pcd.features_padded(), agg_idx)  # (N_batch, N_query, K, 1 + 1 + D_desc)

        elif self.agg_method == "knn":
            agg_results = knn_points(
                query_pcd.points_padded(),
                scene_pcd.points_padded(),
                lengths1=query_pcd.num_points_per_cloud(),
                lengths2=scene_pcd.num_points_per_cloud(),
                K=self.agg_k
            )
            agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
            agg_idx = agg_results.idx  # (N_batch, N_query, K)
            agg_feats = knn_gather(
                scene_pcd.features_padded(),
                agg_idx,
                scene_pcd.num_points_per_cloud()
            )  # (N_batch, N_query, K, 1 + 1 + D_desc)
        else:
            raise NotImplementedError("Other aggregation types not supported")

        # Extract distance embeddings
        N_batch, N_query, K = agg_dists.shape
        agg_dists_flatten = agg_dists.reshape(-1, K, 1)  # (N_batch * N_query, K, 1)

        # Extract semantics embeddings (additionally account for floorplan labels)
        if self.semantics_emb_size != 0:
            semantics_emb = self.semantics_emb(agg_feats[..., 1].long() + 1)  # (N_batch, N_query, K, D_sem_emb)
        else:
            semantics_emb = torch.zeros(N_batch, N_query, K, 0, device=self.device)

        # Extract features including point descriptors
        if self.point_descriptor_size != 0:
            point_desc = agg_feats[..., 2:]  # (N_batch, N_query, K, D_desc)
            point_desc = point_desc.reshape(N_batch, N_query, K, self.point_descriptor_size)

            # Optionally replace floorplan features with learnable ones
            if getattr(self.cfg, "fp_feat_type", "learned") == 'learned':
                fp_mask = (agg_feats[..., 1].long() == -1)
                point_desc[fp_mask] = self.fp_point_desc
        else:
            point_desc = torch.zeros_like(agg_feats[..., 2:])  # (N_batch, N_query, K, D_desc)

        input_emb = torch.cat([point_desc, semantics_emb], dim=-1)  # (N_batch, N_query, K, D_desc + D_sem_emb)

        # Flatten batch and query dimension for inference
        _, _, _, D_input = input_emb.shape
        input_emb_flatten = input_emb.reshape(-1, K, D_input)

        invalid_entry_mask = (agg_idx == -1).reshape(-1, K)  # (N_batch * N_query, K)
        agg_dists_flatten[invalid_entry_mask] = torch.inf  # Infinity values for zeroing out distance weight matrix

        dist_wgt_mtx = torch.exp(-agg_dists_flatten / self.agg_exp_temp)  # (N_batch * N_query, K, 1)

        summary_feats = (dist_wgt_mtx * input_emb_flatten).mean(dim=1)  # (N_batch * N_query, D_input)
        summary_feats = summary_feats.reshape(N_batch, N_query, -1)  # (N_batch, N_query, D_input)

        return summary_feats

    def update_cfg(self, update_dict):
        super(KeyPointSimpleField, self).update_cfg(update_dict)

        # Reset aggregation configs
        self.agg_method = getattr(self.cfg, "agg_method", "ball_query")
        self.agg_k = getattr(self.cfg, "agg_k", 30)
        self.agg_radius = getattr(self.cfg, "agg_radius", 0.5)
        self.agg_radius_limit = getattr(self.cfg, "agg_radius_limit", 30)


class KeyPointMeanField(AbstractFeatureField):
    def __init__(self, cfg, device: torch.device):
        super(KeyPointMeanField, self).__init__(cfg, device)

        # Set aggregation configs
        self.agg_method = getattr(cfg, "agg_method", "ball_query")
        self.agg_k = getattr(cfg, "agg_k", 30)
        self.agg_radius = getattr(cfg, "agg_radius", 0.5)

        # Set point descriptor size
        self.point_descriptor_size = getattr(self.cfg, 'point_descriptor_size', 48)

        # Set semantics embedding
        self.semantics_emb_size = getattr(cfg, 'semantics_emb_size', 32)

        if self.semantics_emb_size != 0:
            # Include additional label for floorplan corners (treated as label 0)
            self.semantics_emb = nn.Embedding(cfg.num_classes + 1, self.semantics_emb_size, device=self.device)

            if not getattr(cfg, 'train_semantics_emb', False):  # Optionally freeze semantics embedding
                for param in self.semantics_emb.parameters():
                    param.requires_grad = False
        else:
            self.semantics_emb = None

        # Optionally initialize learnable point descriptor for floorplan corners
        if self.cfg.fp_feat_type == 'learned':
            self.fp_point_desc = torch.nn.Parameter(torch.randn(self.point_descriptor_size, device=self.device), requires_grad=True)
        else:
            self.fp_point_desc = None

        # Set distance encoding
        self.distance_emb_size = getattr(cfg, 'distance_emb_size', 32)
        if self.cfg.distance_emb_type == 'fixed':
            self.dist_emb = PositionalEncoding(1, self.distance_emb_size // 2)
        elif self.cfg.distance_emb_type == 'direct_learned':
            self.dist_emb = DirectDistanceEmb(self.distance_emb_size).to(self.device)

            # Freeze distance embedding
            for param in self.dist_emb.parameters():
                param.requires_grad = False
        else:  # Learned distance embeddings
            self.dist_emb = DistanceEmb(self.distance_emb_size).to(self.device)

            # Freeze distance embedding
            for param in self.dist_emb.parameters():
                param.requires_grad = False

        self.normalize_feats = getattr(cfg, 'normalize_feats', False)

    def forward(self, query_pcd: Pointclouds, scene_pcd: Pointclouds):
        # Extract neighborhood points (NOTE: features are in order [instance semantics descriptor])
        # NOTE: Distances are by default squared in ball_query, so this needs to be reverted
        if self.agg_method == "ball_query":
            agg_results = ball_query(
                query_pcd.points_padded(),
                scene_pcd.points_padded(),
                lengths1=query_pcd.num_points_per_cloud(),
                lengths2=scene_pcd.num_points_per_cloud(),
                K=adaptive_k_from_radius(self.agg_radius),
                radius=self.agg_radius,
                return_nn=True
            )
            agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
            agg_idx = agg_results.idx  # (N_batch, N_query, K)
            agg_feats = masked_gather(scene_pcd.features_padded(), agg_idx)  # (N_batch, N_query, K, 1 + 1 + D_desc)

        elif self.agg_method == "knn":
            agg_results = knn_points(
                query_pcd.points_padded(),
                scene_pcd.points_padded(),
                lengths1=query_pcd.num_points_per_cloud(),
                lengths2=scene_pcd.num_points_per_cloud(),
                K=self.agg_k
            )
            agg_dists = torch.sqrt(agg_results.dists)  # (N_batch, N_query, K)
            agg_idx = agg_results.idx  # (N_batch, N_query, K)
            agg_feats = knn_gather(
                scene_pcd.features_padded(),
                agg_idx,
                scene_pcd.num_points_per_cloud()
            )  # (N_batch, N_query, K, 1 + 1 + D_desc)
        else:
            raise NotImplementedError("Other aggregation types not supported")

        # Extract distance embeddings
        N_batch, N_query, K = agg_dists.shape
        agg_dists_flatten = agg_dists.reshape(-1, 1) / self.agg_radius  # NOTE: Normalized to ensure range in [0, 1]
        dists_emb = self.dist_emb(agg_dists_flatten)  # (N_batch * N_query * K, D_dist_emb)
        dists_emb = dists_emb.reshape(N_batch, N_query, K, self.distance_emb_size)  # (N_batch, N_query, K, D_dist_emb)

        # Extract semantics embeddings (additionally account for floorplan labels)
        if self.semantics_emb_size != 0:
            semantics_emb = self.semantics_emb(agg_feats[..., 1].long() + 1)  # (N_batch, N_query, K, D_sem_emb)
        else:
            semantics_emb = torch.zeros(N_batch, N_query, K, 0, device=self.device)

        # Extract features including point descriptors
        if self.point_descriptor_size != 0:
            point_desc = agg_feats[..., 2:]  # (N_batch, N_query, K, D_desc)
            point_desc = point_desc.reshape(N_batch, N_query, K, self.point_descriptor_size)

            # Optionally replace floorplan features with learnable ones
            if getattr(self.cfg, "fp_feat_type", "learned") == 'learned':
                fp_mask = (agg_feats[..., 1].long() == -1)
                point_desc[fp_mask] = self.fp_point_desc
        else:
            point_desc = torch.zeros_like(agg_feats[..., 2:])  # (N_batch, N_query, K, D_desc)

        input_emb = torch.cat([point_desc, dists_emb, semantics_emb], dim=-1)  # (N_batch, N_query, K, D_desc + D_dist_emb + D_sem_emb)

        # Flatten batch and query dimension for inference
        _, _, _, D_input = input_emb.shape
        input_emb_flatten = input_emb.reshape(-1, K, D_input)
        summary_feats = input_emb_flatten.mean(1)
        summary_feats = summary_feats.reshape(N_batch, N_query, -1)  # (N_batch, N_query, D_input)

        # Optionally normalize features
        if self.normalize_feats:
            summary_feats = summary_feats / summary_feats.norm(dim=-1, keepdim=True)

        return summary_feats

    def update_cfg(self, update_dict):
        super(KeyPointMeanField, self).update_cfg(update_dict)

        # Reset aggregation configs
        self.agg_method = getattr(self.cfg, "agg_method", "ball_query")
        self.agg_k = getattr(self.cfg, "agg_k", 30)
        self.agg_radius = getattr(self.cfg, "agg_radius", 0.5)
        self.agg_radius_limit = getattr(self.cfg, "agg_radius_limit", 30)


if __name__ == '__main__':
    # Test script on positional encoding
    emb_size = 16
    test_dist = torch.linspace(0., 1., 1000).reshape(-1, 1)
    encoder = PositionalEncoding(1, 16)
    embedding = encoder(test_dist)

    test_dist = test_dist.reshape(-1).numpy()
    embedding = embedding.numpy()

    for emb_idx in range(embedding.shape[-1]):
        vis_emb = embedding[:, emb_idx]
        plt.subplot(embedding.shape[-1], 1, emb_idx + 1)
        plt.plot(test_dist, vis_emb)
    plt.show()
