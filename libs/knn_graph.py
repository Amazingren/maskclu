import numpy as np
import torch
import torch.nn.functional as F


def compute_morton_codes(x_quantized, y_quantized, z_quantized, n_bits):
    B, N = x_quantized.shape

    # Convert integers to bits
    bits_x = ((x_quantized.unsqueeze(-1) >> torch.arange(n_bits, device=x_quantized.device)) & 1).byte()
    bits_y = ((y_quantized.unsqueeze(-1) >> torch.arange(n_bits, device=y_quantized.device)) & 1).byte()
    bits_z = ((z_quantized.unsqueeze(-1) >> torch.arange(n_bits, device=z_quantized.device)) & 1).byte()

    # Stack and interleave bits to form Morton codes
    bits = torch.stack([bits_x, bits_y, bits_z], dim=-1)  # [B, N, n_bits, 3]
    bits = bits.permute(0, 1, 3, 2).reshape(B, N, -1)  # [B, N, 3 * n_bits]

    # Precompute exponents for bit positions
    exponents = torch.arange(n_bits * 3, device=bits.device).unsqueeze(0).unsqueeze(0)  # [1, 1, n_bits * 3]

    # Calculate Morton codes
    morton_codes = torch.sum(bits.long() * (1 << exponents), dim=-1)  # [B, N]

    return morton_codes  # [B, N]


def label_to_centers(points, labels, is_norm=False):
    centers = torch.einsum('bnd,bnk->bkd', points, labels)

    if is_norm:
        weighs = torch.sum(labels, dim=1).unsqueeze(-1) + 1e-4
        centers = centers / weighs

    return centers


def aggregate_superpoints_to_points(G_t, sp_features):
    """
    Aggregate superpoint features back to points based on the soft association matrix G_t.
    """
    # Perform the aggregation: Multiply G_t (B x N_p x N_sp) with sp_features (B x N_sp x C_sp)
    aggregated_features = torch.bmm(G_t, sp_features)  # Shape: (B, N_p, C_sp)
    return aggregated_features


def normalized_laplacian(adj_matrix, EPS=1e-15):
    # adj_matrix: B x N x N
    B, N, _ = adj_matrix.shape

    # Compute the degree matrix D (sum of rows of adj_matrix)
    degree = torch.sum(adj_matrix, dim=-1)  # B x N

    # Avoid division by zero
    degree_inv_sqrt = 1.0 / torch.sqrt(degree + EPS)
    degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # B x N x N

    # Identity matrix
    identity = torch.eye(N, device=adj_matrix.device).unsqueeze(0).expand(B, N, N)

    # Compute normalized Laplacian: L = I - D^{-1/2} * A * D^{-1/2}
    norm_adj = torch.bmm(torch.bmm(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)
    laplacian = identity - norm_adj

    return laplacian, norm_adj


def normalize_points(points):
    # Normalize coordinates to [0, 1]
    x_min = points.min(dim=1, keepdim=True).values  # [B, 1, 3]
    x_max = points.max(dim=1, keepdim=True).values  # [B, 1, 3]
    points_normalized = (points - x_min) / (x_max - x_min + 1e-6)  # [B, N, 3]
    return points_normalized


def construct_similarity_matrix(points, features, distance_threshold, window_size=10, n_bits=10, gamma=-1,
                                is_zeros=False):
    B, N, _ = points.shape
    _, _, D = features.shape

    # Normalize and quantize points
    max_int = 2 ** n_bits - 1
    points_normalized = normalize_points(points)  # [B, N, 3], normalized to [0,1]
    points_quantized = (points_normalized * max_int).long()  # [B, N, 3]
    x_quantized = points_quantized[:, :, 0]  # [B, N]
    y_quantized = points_quantized[:, :, 1]
    z_quantized = points_quantized[:, :, 2]

    # Compute Morton codes
    morton_codes = compute_morton_codes(x_quantized, y_quantized, z_quantized, n_bits)  # [B, N]

    # Sort points and features according to Morton codes
    sorted_morton_codes, indices = torch.sort(morton_codes, dim=1)  # [B, N], [B, N]
    points_sorted = points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, N, 3]
    features_sorted = features.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]

    # Build neighbor indices using a sliding window
    # You can adjust this value based on your datasets
    neighbor_offsets = torch.arange(-window_size, window_size + 1, device=points.device)  # [2 * window_size + 1]
    K = neighbor_offsets.shape[0]  # Total number of neighbors considered per point
    indices_i = torch.arange(N, device=points.device).unsqueeze(-1) + neighbor_offsets  # [N, K]
    indices_i = indices_i.clamp(0, N - 1)  # [N, K]
    indices_i = indices_i.unsqueeze(0).expand(B, -1, -1)  # [B, N, K]

    # Get neighbor indices
    neighbor_indices = indices_i  # Indices in sorted order
    # Get neighbor points and features
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, N, K)  # [B, N, K]
    neighbor_points = points_sorted[batch_indices, neighbor_indices]  # [B, N, K, 3]
    neighbor_features = features_sorted[batch_indices, neighbor_indices]  # [B, N, K, D]

    # Compute distances to neighbor points
    points_expanded = points_sorted.unsqueeze(2)  # [B, N, 1, 3]
    distances = torch.norm(points_expanded - neighbor_points, dim=-1)  # [B, N, K]

    # Create a mask where distances are within the threshold
    within_threshold = distances <= distance_threshold  # [B, N, K]

    # Compute feature similarity between points and neighbor points
    if gamma > 0:
        pairwise_distances = torch.norm(features_sorted.unsqueeze(2) - neighbor_features, dim=-1)
        # Step 2: Apply the RBF (Gaussian) kernel to the pairwise distances
        feature_similarity = torch.exp(-gamma * pairwise_distances)  # Shape: [B, N, K]
    else:
        # Normalize features for cosine similarity
        features_normalized = F.normalize(features_sorted, dim=-1)  # [B, N, D]
        neighbor_features_normalized = F.normalize(neighbor_features, dim=-1)  # [B, N, K, D]
        # Compute dot product along the feature dimension
        feature_similarity = 1e-4 + torch.sum(
            features_normalized.unsqueeze(2) * neighbor_features_normalized, dim=-1
        )  # [B, N, K]
    # Get indices and mask
    i_indices = torch.arange(N, device=points.device).view(1, N, 1).expand(B, -1, K)  # [B, N, K]
    j_indices = neighbor_indices  # [B, N, K]
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, N, K)  # [B, N, K]

    # Flatten the indices and values where within_threshold is True
    mask = within_threshold  # [B, N, K]
    batch_indices_flat = batch_indices[mask]  # [num_pairs]
    i_indices_flat = i_indices[mask]  # [num_pairs]
    j_indices_flat = j_indices[mask]  # [num_pairs]
    values_flat = feature_similarity[mask]  # [num_pairs]

    # Initialize the similarity matrix A
    adj = torch.zeros(B, N, N, device=points.device)

    # Use scatter_add to handle potential duplicate indices
    adj[batch_indices_flat, i_indices_flat, j_indices_flat] = values_flat

    if is_zeros:
        # Ensure the diagonal elements are zero (no self-loops)
        eye = torch.eye(N).unsqueeze(0).to(adj.device)  # Identity matrix for each batch
        adj = adj * (1 - eye)  # Zero out diagonal

    return adj


def knn_based_affinity_matrix(xyz, feats, k=10, gamma=-1, is_zeros=True):
    """
    Compute affinity matrix based on the k-nearest neighbors (KNN).

    Args:
        xyz (torch.Tensor): Point cloud coordinates of shape [B, N, 3].
        feats (torch.Tensor): Point cloud features of shape [B, N, D].
        k (int): Number of nearest neighbors to consider for each point.
        gamma (float): Optional RBF kernel parameter. If gamma > 0, use RBF kernel instead of cosine similarity.
        is_zeros (bool): If True, sets the diagonal of the affinity matrix to 0 (no self-loops).

    Returns:
        torch.Tensor: Affinity matrix of shape [B, N, N], where entries represent similarities
                      between each point and its k-nearest neighbors.
    """
    B, N, _ = xyz.shape

    # Step 1: Compute pairwise Euclidean distances between points
    dist_matrix = torch.cdist(xyz, xyz, p=2)  # Shape: [B, N, N]

    # Step 2: Find the indices of the k-nearest neighbors for each point
    knn_indices = dist_matrix.topk(k=k, dim=-1, largest=False).indices  # Shape: [B, N, k]

    # Step 3: Compute cosine similarity or RBF kernel for point features
    # Normalize features to compute cosine similarity
    feats_normalized = F.normalize(feats, p=2, dim=-1)  # Shape: [B, N, D]
    if gamma > 0:
        # Compute pairwise Euclidean distances for features
        pairwise_feat_dist = torch.cdist(feats_normalized, feats_normalized, p=2)  # Shape: [B, N, N]
        # Apply RBF (Gaussian) kernel
        affinity_scores = torch.exp(-gamma * pairwise_feat_dist)  # Shape: [B, N, N]
    else:
        # Compute cosine similarity between features
        affinity_scores = torch.matmul(feats_normalized, feats_normalized.transpose(1, 2))  # Shape: [B, N, N]
        affinity_scores = 1 + affinity_scores

    # Step 4: Create the affinity matrix using only k-nearest neighbors
    knn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)  # Shape: [B, N, N]
    knn_mask.scatter_(2, knn_indices, True)  # Set True for the k-nearest neighbors
    affinity_matrix = torch.where(knn_mask, affinity_scores, torch.zeros_like(affinity_scores))  # Mask
    affinity_dist = torch.exp(dist_matrix.mean(dim=(1, 2), keepdim=True) - dist_matrix)
    affinity_dist = torch.where(knn_mask, affinity_dist, torch.zeros_like(affinity_dist))  # Mask
    affinity_matrix = affinity_dist * affinity_matrix
    # Step 5: Symmetrize the affinity matrix
    affinity_matrix = (affinity_matrix + affinity_matrix.transpose(-1, -2)) / 2.0

    if is_zeros:
        # Ensure the diagonal elements are zero (no self-loops)
        eye = torch.eye(N).unsqueeze(0).to(affinity_matrix.device)  # Identity matrix for each batch
        affinity_matrix = affinity_matrix * (1 - eye)  # Zero out diagonal

    return affinity_matrix


if __name__ == '__main__':
    # Example usage:
    B, N, D = 2, 10000, 64  # Batch size, number of points, feature dimension
    points = torch.rand(B, N, 3).cuda()  # Random point cloud datasets
    features = torch.rand(B, N, D).cuda()  # Random features
    distance_threshold = 0.1  # Set your desired distance threshold

    A = construct_similarity_matrix(points, features, distance_threshold)
    print(A.shape)  # Should be [B, N, N]
