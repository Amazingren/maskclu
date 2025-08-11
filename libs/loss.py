import numpy as np
import torch
from torch.nn import functional as F

from libs.lib_utils import assignment, sinkhorn, gmm_params


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1)
    return loss.mean()


def contrastive_loss_3d(k, q, tau=0.1, is_norm=False):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    - k (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - q (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    - torch.Tensor: Scalar tensor representing the loss.
    """
    b, n, d = k.shape  # Assuming k and q have the same shape [b, n, d]

    # Normalize k and q along the feature dimension
    if is_norm:
        k = F.normalize(k, dim=2)
        q = F.normalize(q, dim=2)
        logits = torch.einsum('bnd,bmd->bnm', [k, q]) / tau
    # Compute cosine similarity as dot product of k and q across all pairs
    logits = torch.einsum('bnd,bmd->bnm', [k, q]) / np.sqrt(d)

    # Create labels for positive pairs: each sample matches with its corresponding one in the other set
    labels = torch.arange(n).repeat(b).to(logits.device)
    labels = labels.view(b, n)
    # Use log_softmax for numerical stability and compute the cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


def cut_loss(adj, s, mask=None, alpha=-1):
    """_summary_

    Args:
        adj (_type_): Normalized adjacent matrix of shape (B x N x N).
        s (_type_): Cluster assignment score matrix of shape (B x N x K).
        mask (_type_, optional): Mask matrix
            :math:`\mathbf{M} \in \{0, 1\}^{B \times N}` indicating
            the valid nodes for each graph. Defaults to :obj:`None`.
        alpha (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: the MinCut loss, and the orthogonality loss.
    """
    # Ensure batch dimension
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, k = s.size()
    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(s.dtype)
        s = s * mask

    # Pool node features and adjacency matrix
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)  # Shape: [B, C, C]

    # MinCut regularization
    mincut_num = torch.einsum('bii->b', out_adj)  # Trace of out_adj
    d_flat = adj.sum(dim=-1)  # Degree matrix diagonal, Shape: [B, N]
    d = torch.diag_embed(d_flat)  # Shape: [B, N, N]
    mincut_den = torch.einsum('bii->b', torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = 1 - (mincut_num / (mincut_den + 1e-10))
    mincut_loss = mincut_loss.mean()

    if alpha > 0:
        # Orthogonality regularization
        ss = torch.matmul(s.transpose(1, 2), s)  # Shape: [B, C, C]
        ss_fro = torch.norm(ss, dim=(1, 2), keepdim=True)  # Frobenius norm, Shape: [B, 1, 1]
        i_s = torch.eye(k, device=s.device).unsqueeze(0)  # Identity matrix, Shape: [1, C, C]
        i_s = i_s / torch.sqrt(torch.tensor(k, dtype=s.dtype, device=s.device))  # Normalize
        ortho_loss = torch.norm(ss / ss_fro - i_s, dim=(1, 2))
        ortho_loss = ortho_loss.mean()
        return mincut_loss + alpha * ortho_loss

    return mincut_loss


def update_centers(x, centroids):
    bs, num, _ = x.shape
    # Compute distances and find the closest centroids (hard assignment)
    dists = torch.cdist(x, centroids)  # Compute the pairwise distances
    min_indices = torch.argmin(dists, dim=-1)  # Get the index of the closest centroid

    # Create the hard assignment matrix
    gamma = torch.zeros_like(dists)  # Same shape as `dists`
    gamma[torch.arange(bs).unsqueeze(1), torch.arange(num), min_indices] = 1

    # Avoid division by zero
    gamma = gamma / torch.sum(gamma, dim=1, keepdim=True).clip(min=1e-4)

    # Update centroids based on hard assignment
    centroids = torch.einsum('bmk,bmd->bkd', gamma, x)

    return centroids


def welsch_loss(pts, centers, sigma=1.0):
    """
    Computes the Welsch loss between the input and the target.

    Args:
        pts (Tensor): Predicted values, shape (B, N, D) or (B, D).
        centers (Tensor): Ground truth values, same shape as input.
        sigma (float): Scale parameter controlling the shape of the loss.

    Returns:
        Tensor: The average Welsch loss.
    """
    # Compute the element-wise residuals
    residual = torch.cdist(pts, centers).min(dim=-1)[0]  # Shape: (B, N, D) or (B, D)

    # Compute the squared residuals
    squared_residual = torch.sum(residual ** 2, dim=-1)  # Sum over feature dimensions

    # Apply the Welsch loss function element-wise
    loss = 1 - torch.exp(-squared_residual / (2 * sigma ** 2))
    n = pts.size(1)
    k = centers.size(1)
    loss = k * loss.sum(dim=-1).mean() / n

    # Average the loss across all elements
    return loss


def align_loss_3d(q_features, k_features, centers, tau=0.2, sink_tau=0.1):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    - k (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - q (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    - torch.Tensor: Scalar tensor representing the loss.
    """
    k_features = F.normalize(k_features, dim=-1)
    centers = F.normalize(centers, dim=-1)
    q_features = F.normalize(q_features, dim=-1)
    with torch.no_grad():
        # Compute cosine similarity as dot product of k and q across all pairs
        gamma = assignment(k_features, centers, iters=10, tau=sink_tau)
        gamma = gamma / torch.sum(gamma, dim=-1, keepdim=True).clip(min=1e-3)
    # Use log_softmax for numerical stability and compute the cross-entropy loss
    logits = torch.einsum('bnd,kd->bnk', q_features, centers)
    loss = -torch.sum(gamma.detach() * torch.log_softmax(logits / tau, dim=-1), dim=-1).mean()
    n_matrix = torch.matmul(centers, centers.transpose(1, 0))
    device = n_matrix.device
    n_loss = F.mse_loss(torch.eye(n_matrix.size(0)).to(device), n_matrix)

    return loss + 0.1 * n_loss


def contrastive_loss_2d(k, q, tau=0.1, is_norm=False):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    - k (torch.Tensor): Tensor of shape [b, d] containing a batch of embeddings.
    - q (torch.Tensor): Tensor of shape [b, d] containing a batch of embeddings.
    - tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    - torch.Tensor: Scalar tensor representing the loss.
    """
    b = k.size(0)
    # Normalize k and q along the feature dimension
    if is_norm:
        k = F.normalize(k, dim=1)
        q = F.normalize(q, dim=1)
    # Compute cosine similarity as dot product of k and q across all pairs
    logits = torch.einsum('md,nd->mn', [k, q]) / tau

    # Create labels for positive pairs: each sample matches with its corresponding one in the other set
    labels = torch.arange(b).to(logits.device)
    labels = labels.view(-1)
    # Use log_softmax for numerical stability and compute the cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


def soft_kmeans_loss_from_subset(points, sub_points, sub_assignment_matrix, temperature=1.0):
    """
    Computes the soft k-means loss for all points using cluster centers computed from a subset.

    Args:
        points (torch.Tensor): Full set of points, shape (B, N, D)
        sub_points (torch.Tensor): Subset of points, shape (B, M, D)
        sub_assignment_matrix (torch.Tensor): Assignment matrix for sub_points, shape (B, M, K)
        temperature (float): Temperature parameter for soft assignments (default is 1.0)

    Returns:
        torch.Tensor: The computed soft k-means loss (scalar)
    """
    B, N, D = points.shape
    _, _, K = sub_assignment_matrix.shape

    # Step 1: Compute cluster centers from the subset
    cluster_weights, cluster_centers = gmm_params(sub_assignment_matrix, sub_points)  # Shape: (B, K, D)

    # Step 2: Compute squared Euclidean distances between all points and cluster centers
    sq_distances = torch.cdist(points, cluster_centers)

    # Step 3: Compute assignment logits including cluster weights
    log_cluster_weights = torch.log(cluster_weights.unsqueeze(1) + 1e-8)  # Shape: (B, 1, K)
    assignment_logits = log_cluster_weights - sq_distances / temperature  # Shape: (B, N, K)

    # Step 4: Compute soft assignments based on assignment logits
    assignment_matrix = torch.softmax(assignment_logits, dim=-1)  # Shape: (B, N, K)

    # Step 5: Compute the weighted squared distances
    weighted_sq_distances = assignment_matrix * sq_distances  # Shape: (B, N, K)

    # Step 6: Compute the loss by averaging over all points and clusters
    loss_per_sample = weighted_sq_distances.sum(dim=(1, 2)) / N  # Shape: (B,)
    soft_kmeans_loss = loss_per_sample.mean()  # Scalar

    return soft_kmeans_loss


def spatial_loss(cts_shift, c_cts, s, tau=0.1):
    # Compute global loss component (requires definition of welsch_loss)
    gamma = sinkhorn(-torch.cdist(cts_shift, c_cts) / tau)
    gamma = gamma / torch.sum(gamma, dim=1, keepdim=True).clip(min=1e-4)
    ass_loss = -torch.sum(gamma.detach() * torch.log_softmax(s, dim=-1), dim=-1).sum(dim=-1).mean()
    b, m, n = s.size()
    return n * ass_loss / m


def info_nce_loss(features, centers, assign_matrix, temperature=1.0):
    """
    Computes the InfoNCE loss for a batch of features against given centers.

    Args:
        centers (torch.Tensor): The centers (prototypes) of shape [B, K, D].
        assign_matrix (torch.Tensor): The assignment matrix of shape [B, N, K].
        features (torch.Tensor): The feature vectors of shape [B, N, D].
        temperature (float): Temperature parameter to scale the logits.

    Returns:
        torch.Tensor: The computed InfoNCE loss.
    """
    B, K, D = centers.shape

    # Step 1: Compute the similarity (dot product) between each feature and each center
    # Resulting shape will be [B, N, K]
    logits = torch.bmm(features, centers.transpose(1, 2)) / np.sqrt(D)  # Shape: [B, N, K]

    # Step 2: Apply log softmax across the K centers (class dimension)
    log_probs = F.log_softmax(logits / temperature, dim=-1)  # Shape: [B, N, K]

    # Step 3: Compute the InfoNCE loss
    # Multiply log probabilities by the assignment matrix and sum over classes (K dimension)
    loss_per_sample = -(assign_matrix * log_probs).sum(dim=-1)  # Shape: [B, N]

    # Average over all samples and batches
    loss = loss_per_sample.mean()  # Scalar

    return loss


if __name__ == '__main__':
    # Assuming you have your data
    B, N, D = 2, 1000, 3  # Batch size, number of points, dimensionality
    K = 5  # Number of clusters
    M = 200  # Size of the subset

    # Generate random data for demonstration
    points = torch.randn(B, N, D)
    sub_points = points[:, :M, :]  # Use the first M points as the subset

    # Simulate an assignment matrix for the subset (e.g., from a model's output)
    sub_assignment_logits = torch.randn(B, M, K)
    sub_assignment_matrix = torch.softmax(sub_assignment_logits, dim=-1)

    # Compute the loss
    loss = soft_kmeans_loss_from_subset(points, sub_points, sub_assignment_matrix)

    print("Soft k-means loss:", loss.item())
