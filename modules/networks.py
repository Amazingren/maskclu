from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from libs.knn_graph import knn_based_affinity_matrix, normalized_laplacian
from libs.lib_utils import index_points, knn, KNN, fps


def fill_empty_indices(idx: torch.Tensor) -> torch.Tensor:
    """
    replaces all empty indices (-1) with the first index from its group
    """
    B, G, K = idx.shape

    mask = idx == -1
    first_idx = idx[:, :, 0].unsqueeze(-1).expand(-1, -1, K)
    idx[mask] = first_idx[mask]  # replace -1 index with first index
    # print(f"DEBUG: {(len(idx[mask].view(-1)) / len(idx.view(-1))) * 100:.1f}% of ball query indices are empty")

    return idx


def get_embedding_indices(points, sigma_d=0.5, sigma_e=0.1, angle_k=3, reduction_a='max'):
    r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.
    Args:
        points: torch.Tensor (B, N, 3), input point cloud
    Returns:
        d_indices: torch.FloatTensor (B, N, N), distance embedding indices
        a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
    """
    batch_size, num_point, _ = points.shape
    dist_map = torch.cdist(points, points)  # (B, N, N)

    # d_indices = (dist_map / (1.0+dist_map))
    d_indices = dist_map / sigma_d
    # Check for NaNs
    k = angle_k
    knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
    knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
    expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
    knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
    ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
    anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
    ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
    anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
    sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
    cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
    angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
    a_indices = angles
    if reduction_a == 'max':
        a_indices = a_indices.max(dim=-1)[0]
    else:
        a_indices = a_indices.mean(dim=-1)

    return torch.cat([d_indices.unsqueeze(-1), a_indices.unsqueeze(-1) / sigma_e], dim=-1)


class MinCutPoolingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, temp=1.0):
        super(MinCutPoolingLayer, self).__init__()

        self.Wm = nn.Linear(input_dim, hidden_dim, bias=False)  # Mixing weights
        self.Ws = nn.Linear(input_dim, hidden_dim, bias=False)  # Skip connection weights
        self.W1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(input_dim)
        self.W2 = nn.Linear(hidden_dim, K)  # K is the number of clusters
        self.temp = temp

    def forward(self, node_fea, adj):
        """
        node_fea: Node feature matrix (B x N x F)
        adj: Symmetrically normalized adjacency matrix (B x N x N)
        """
        # Message Passing operation: ReLU(adj X Wm) + X Ws
        B, N, _ = node_fea.size()
        reg = torch.eye(N, device=adj.device).unsqueeze(0).expand_as(adj)
        out = torch.relu(torch.bmm(adj + reg, self.Wm(node_fea))) + self.Ws(node_fea)
        s = self.W2(torch.relu(self.W1(out))) / self.temp
        # Compute the cluster assignment matrix S (B x N x K)
        gamma = torch.softmax(s, dim=-1)

        # Compute the pooled feature matrix X' (B x K x F)
        node_pool = torch.bmm(gamma.transpose(1, 2), node_fea)
        node_pool = self.norm(node_pool)

        return node_pool, gamma, s


class PosE(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1000, beta=100):
        super(PosE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta

    def forward(self, xyz):
        B, N, _ = xyz.shape
        device = xyz.device  # Get the device of the input tensor (CPU/GPU)
        feat_dim = self.out_dim // (self.in_dim * 2)

        # Create feat_range on the same device as xyz
        feat_range = torch.arange(feat_dim, dtype=torch.float32, device=device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)

        # Stack the embeddings and reshape
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.reshape(B, N, self.out_dim)

        return position_embed


def get_graph_feature(x, idx=None, k=8):
    """
    Computes the k-nearest neighbor graph features for input tensor x using torch.gather.

    Args:
        idx:
        x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_points).
        k (int): Number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Tensor containing the graph features of shape
                      (batch_size, 2 * num_dims, num_points, k).
    """
    batch_size, num_dims, num_points = x.size()
    # Transpose to (batch_size, num_points, num_dims)
    x = x.transpose(2, 1).contiguous()
    # Get k-nearest neighbor indices for each point
    if idx is None:
        idx = knn(x, x, k=k)  # Shape: (batch_size, num_points, k)

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class TranFeat(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TranFeat, self).__init__(*args, **kwargs)

    def forward(self, x, y=None):
        if y is None:
            return x.transpose(2, 1).contiguous()
        else:
            return x.transpose(2, 1).contiguous(), y.transpose(2, 1).contiguous()


class TDConv(nn.Module):
    def __init__(self, input_dim, dim, k=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2))
        self.k = k

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1)[0]

        return x


def compute_lra_one(group_xyz, weighting=False):
    """
    Compute the Local Reference Axis (LRA) for a group of points.

    Args:
        group_xyz (torch.Tensor): Tensor of shape [B, G, N, 3] containing point coordinates.
        weighting (bool): Whether to apply distance-based weighting.

    Returns:
        torch.Tensor: Local Reference Axis (LRA) of shape [B, G, 3].
    """
    # Compute the Euclidean distances (norms) of the points
    dists = torch.norm(group_xyz, dim=-1, keepdim=True)  # Shape: [B, G, N, 1]

    if weighting:
        # Apply distance-based weighting
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True) + 1e-4
        weights = dists / dists_sum
        weights[weights != weights] = 1.0  # Handle NaNs
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz) + 1e-4
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz) + 1e-4

    # Compute eigenvalues and eigenvectors using torch.linalg.eigh
    eigen_values, vec = torch.linalg.eigh(M)

    # Use the first eigenvector as the LRA
    lra = vec[:, :, :, 0]

    # Normalize the LRA
    lra_length = torch.norm(lra, dim=-1, keepdim=True) + 1e-4
    lra = lra / lra_length
    return lra  # Shape: [B, G, 3]




# class Encoder(nn.Module):
#     """
#     Encoder Module for embedding point groups with attention and relative position encoding.
#     """
#
#     def __init__(self, encoder_channel):
#         super(Encoder, self).__init__()
#         self.encoder_channel = encoder_channel
#
#         # Initial convolution layers for encoding point coordinates and LRA
#         self.first_conv = nn.Sequential(
#             nn.Conv1d(6, 128, 1),  # Input is [xyz, lra] (6 channels)
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 256, 1)
#         )
#
#         # Second convolution layers to increase feature dimensionality
#         self.second_conv = nn.Sequential(
#             nn.Conv1d(512, 512, 1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(512, self.encoder_channel, 1)
#         )
#
#     def forward(self, point_groups):
#         """
#         Forward pass of the Encoder.
#
#         Args:
#             point_groups (torch.Tensor): Input tensor of shape [B, G, N, 3], where
#                 B - Batch size
#                 G - Number of groups
#                 N - Number of points in each group
#                 3 - Coordinates of each point
#
#         Returns:
#             torch.Tensor: Output tensor of shape [B, G, C], where C is the encoder channel.
#         """
#         bs, g, n, _ = point_groups.shape
#
#         # Compute the Local Reference Axis (LRA) for each group
#         lra = compute_lra_one(point_groups, weighting=True)  # Shape: [B, G, 3]
#         lra = lra.view(bs * g, 1, 3).expand(-1, n, -1)  # Expand LRA to match point dimensions
#
#         # Reshape point groups for processing
#         point_groups = point_groups.view(bs * g, n, 3)
#
#         # Concatenate point coordinates with LRA as additional channels
#         rel_point_groups = torch.cat([point_groups, lra], dim=-1)  # Shape: [BG, N, 6]
#
#         # Encode point features with the first convolution layer
#         feature = self.first_conv(rel_point_groups.transpose(2, 1))  # Shape: [BG, 256, N]
#         feature_global = torch.max(feature, dim=-1)[0].unsqueeze(-1)  # Shape: [BG, 256, 1]
#
#         # Concatenate global and local features, then apply second convolution
#         feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # Shape: [BG, 512, N]
#         feature = self.second_conv(feature)  # Shape: [BG, encoder_channel, N]
#
#         # Pooling to get the global feature for each group
#         feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # Shape: [BG, encoder_channel]
#
#         # Reshape back to [B, G, encoder_channel]
#         return feature_global.view(bs, g, self.encoder_channel)


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(4, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = torch.cat([point_groups, torch.norm(point_groups, dim=-1, keepdim=True)], dim=-1)
        feature = self.first_conv(feature.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(center, xyz)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class PointNetFeatureUpsampling(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position datasets, [B, C, N]
            xyz2: sampled input points position datasets, [B, C, S]
            points1: input points datasets, [B, D, N]
            points2: input points datasets, [B, D, S]
        Return:
            new_points: upsampled points datasets, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = torch.cdist(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dists[dists < 0] = 0
            dist_recip = 1.0 / (dists + torch.finfo(dists.dtype).eps)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points.transpose(1, 2))).transpose(1, 2))
        return new_points


class PointMasking(nn.Module):
    def __init__(self, ratio: float, type: str):
        super().__init__()
        self.ratio = ratio

        if type == "rand":
            self.forward = self._mask_center_rand
        elif type == "block":
            self.forward = self._mask_center_block
        else:
            raise ValueError(f"No such masking type: {type}")

    def _mask_center_rand(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Randomly masks a proportion of centers without using a for loop.

        Args:
            centers: Tensor of shape (B, G, 3), where B is the batch size, G is the number of centers.

        Returns:
            mask: Boolean tensor of shape (B, G), where True represents a masked center.
        """
        # If ratio is zero, return a mask of all False (no masking)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2], dtype=torch.bool, device=centers.device)

        B, G, _ = centers.shape
        num_mask = int(self.ratio * G)

        # Create base mask of shape (G,) with num_mask True values
        base_mask = torch.cat([
            torch.zeros(G - num_mask, device=centers.device),
            torch.ones(num_mask, device=centers.device)
        ]).to(torch.bool)  # (G,)

        # Generate random permutations for each batch without a loop
        rand_indices = torch.rand(B, G, device=centers.device).argsort(dim=-1)  # (B, G)

        # Expand the base mask and shuffle based on random permutations
        mask = base_mask.unsqueeze(0).expand(B, -1)  # (B, G)
        mask = torch.gather(mask, 1, rand_indices)  # Shuffle the mask for each batch

        return mask

    def _mask_center_block(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Randomly masks a block of nearest neighbors around a randomly selected center for each batch.

        Args:
            centers: Tensor of shape (B, G, 3), where B is the batch size, G is the number of centers, and D (3) is the dimension.

        Returns:
            mask: Boolean tensor of shape (B, G), where True represents the masked points.
        """
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2], dtype=torch.bool, device=centers.device)

        B, G, D = centers.shape
        assert D == 3  # Ensure the input has the correct dimensions

        num_mask = int(self.ratio * G)

        # Randomly select a center for each batch without a loop
        rand_idx = torch.randint(0, G, (B, 1), device=centers.device)  # Random index for each batch
        center = torch.gather(centers, 1, rand_idx.unsqueeze(-1).expand(B, 1, D))  # (B, 1, 3)

        # Find the nearest neighbors of the selected centers
        knn_idx = knn(center.float(), centers.float(), num_mask)  # (B, num_mask)

        # Create the mask and mark the knn indices
        mask = torch.zeros([B, G], device=centers.device)
        mask.scatter_(dim=1, index=knn_idx, value=1.0)
        mask = mask.to(torch.bool)

        return mask


class PointTokenizer(nn.Module):
    def __init__(
            self,
            num_groups: int,
            group_size: int,
            token_dim: int,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.grouping = Group(num_group=num_groups, group_size=group_size
                              )
        self.embedding = Encoder(3, token_dim)

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, 3)
        group: torch.Tensor
        group_center: torch.Tensor
        tokens: torch.Tensor

        group, group_center = self.grouping(points)  # (B, G, K, C), (B, G, 3)
        B, G, S, C = group.shape
        tokens = self.embedding(group.reshape(B * G, S, C)).reshape(
            B, G, self.token_dim
        )  # (B, G, C')
        return tokens, group_center


def get_pos_embed(embed_dim, ipt_pos):
    """
    embed_dim: output dimension for each position
    ipt_pos: [B, G, 3], where 3 is (x, y, z)
    """
    B, G, _ = ipt_pos.size()
    assert embed_dim % 6 == 0
    omega = torch.arange(embed_dim // 6).float().to(ipt_pos.device)  # NOTE
    omega /= embed_dim / 6.
    # (0-31) / 32
    omega = 1. / 10000 ** omega  # (D/6,)
    rpe = []
    for i in range(_):
        pos_i = ipt_pos[:, :, i]  # (B, G)
        out = torch.einsum('bg, d->bgd', pos_i, omega)  # (B, G, D/6), outer product
        emb_sin = torch.sin(out)  # (M, D/6)
        emb_cos = torch.cos(out)  # (M, D/6)
        rpe.append(emb_sin)
        rpe.append(emb_cos)
    return torch.cat(rpe, dim=-1)


if __name__ == '__main__':
    # Assuming x is your input tensor and knn is properly defined
    # x = torch.randn(4, 3, 64)
    # features = get_graph_feature(x, k=8)
    # Example point cloud and features
    B, N, D = 32, 128, 64  # Batch size, number of points, feature dimension
    points = torch.randn(B, N, 3)
    features = torch.randn(B, N, D)

