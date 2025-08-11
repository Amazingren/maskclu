import torch
import torch.nn.functional as F

from libs.lib_utils import farthest_point_sample, index_points, sinkhorn_rpm, gmm_params


def wkmeans(x, num_clusters, dst='feats', iters=10, in_iters=5):
    bs, num, dim = x.shape
    ids = torch.randperm(num)[:num_clusters]
    centroids = x[:, ids, :]
    gamma, pi = torch.zeros((bs, num, num_clusters), requires_grad=True).to(x), None
    for i in range(iters):
        if dst == 'eu':
            log_gamma = - torch.cdist(x, centroids)
        else:
            x = F.normalize(x, p=2, dim=-1)
            centroids = F.normalize(centroids, p=2, dim=-1)
            log_gamma = torch.einsum('bnd,bmd->bnm', x, centroids)
        gamma = sinkhorn_rpm(log_gamma, n_iters=in_iters)
        pi, centroids = gmm_params(gamma, x)

    return gamma, pi, centroids


def self_balanced_min_cut_batch(adjs, n_clus, feats=None, rho=1.5, max_iter=100, in_iter=100, tol=1e-6, soft=-1):
    """
    Performs self-balanced min-cut clustering on a batch of adjacency matrices across multiple GPUs.

    Parameters:
    - adjs (Tensor): Adjacency matrices of shape [B, N, N], where B is the batch size, and N is the number of nodes.
    - n_clus (int): Number of clusters.
    - feats (Tensor, optional): Feature matrix of shape [B, N, D]. If provided, used for initialization.
    - rho (float): Parameter for updating mu.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.
    - soft (bool): If True, returns soft cluster assignments.

    Returns:
    - Tensor: Cluster assignments of shape [B, N] or soft assignments of shape [B, N, n_clus].
    """

    B, N, _ = adjs.size()  # Batch size and number of nodes per batch
    device = adjs.device

    # Initialize cluster assignments Y
    if feats is None:
        # Random initialization
        cls = torch.rand(B, N, n_clus, device=device)
        cls = F.one_hot(torch.argmax(cls, dim=-1), num_classes=n_clus).float()
    else:
        # Initialize using farthest point sampling or KMeans (or other suitable multi-GPU feature sampling)
        ids = farthest_point_sample(feats, n_clus, is_center=False)
        cents = index_points(feats, ids)
        cls = F.one_hot(torch.argmax(-torch.cdist(feats, cents), dim=-1), num_classes=n_clus).float()

    # Precompute matrices
    diag = torch.diag_embed(torch.sum(adjs, dim=-1))  # Degree matrix [B, N, N]
    lap = diag - adjs  # Laplacian matrix [B, N, N]

    # Main optimization loop
    for iteration in range(max_iter):
        # Compute s for each batch
        numerator = torch.einsum('bni,bnm,bmi->b', cls, lap, cls)
        denominator = torch.einsum('bni,bni->b', cls, cls)
        s = (numerator / denominator.clamp(min=1e-8)).view(B, 1, 1)  # Shape: [B, 1, 1]
        # Compute Theta for each batch
        theta = (s / 2) * torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1) - adjs  # [B, N, N]
        # Initialize dual variables
        mu = torch.ones(B, 1, 1, device=device)
        gamma = torch.zeros_like(cls, device=device)

        # Inner loop
        for inner_iter in range(in_iter):
            # Update G
            theta_y = torch.matmul(theta, cls)  # [B, N, n_clus]
            G = cls - (1 / mu) * (theta_y - gamma)
            # Update Y
            if soft > 0:
                cls_new = F.gumbel_softmax(G / soft, hard=True, dim=-1)
            else:
                cls_new = F.one_hot(torch.argmax(G, dim=-1), num_classes=n_clus).float()

            # Check for convergence
            delta = torch.norm(cls_new - cls)
            if delta < tol:
                break
            cls = cls_new
            # Update gamma and mu
            gamma = gamma + mu * (cls - G)
            mu = rho * mu

        # Check for convergence of outer loop
        delta_outer = torch.norm(cls - G)
        if delta_outer < tol:
            print(f"Converged at iteration {iteration}")
            break

    if soft > 0:
        # Return soft assignments (probabilities)
        return F.gumbel_softmax(G / soft, hard=True, dim=-1)  # Note: G represents the continuous assignment here
    return F.one_hot(torch.argmax(cls, dim=-1), num_classes=n_clus).float()


def spectrum_clustering(adj_batch, n_clusters=10, k=2, dst='eu', eps=1e-05, n_iters=20, in_iters=32):
    """
    Perform clustering on a batch of adjacency matrices.
    Args:
        in_iters:
        n_iters:
        eps:
        dst:
        k:
        n_clusters:
        adj_batch (Tensor): A tensor of shape [B, M, M] where B is the batch size.
    Returns:
        Tensor: Cluster labels for each instance in the batch.
    """
    # Threshold the adjacency matrices
    # adj_batch = (adj_batch < self.threshold).float()
    # Calculate the degree matrix D and the Laplacian for each graph in the batch
    diag_batch = torch.sum(adj_batch, dim=2).diag_embed().clip(min=eps)
    laplacian_batch = diag_batch - adj_batch

    # Compute the symmetric normalized Laplacian
    inv_sqrt_diag = torch.diag_embed(torch.pow(torch.diagonal(diag_batch, dim1=-2, dim2=-1), -0.5))
    sym_laplacian_batch = inv_sqrt_diag.bmm(laplacian_batch).bmm(inv_sqrt_diag + eps)
    # Compute eigendecomposition
    e, v = torch.linalg.eigh(sym_laplacian_batch + eps)  # eigh is used for symmetric matrices
    _, idx = torch.topk(e, k, dim=-1, largest=False)

    # Select the k smallest eigenvectors and normalize
    batch_select_v = torch.gather(v, 2, idx.unsqueeze(1).expand(-1, v.size(1), -1))
    norm_v = batch_select_v.div(batch_select_v.norm(p=2, dim=1, keepdim=True).expand_as(batch_select_v) + eps)
    # Apply k-means (example: using a placeholder function `batch_kmeans`)
    gamma = wkmeans(norm_v, n_clusters, dst, iters=n_iters, in_iters=in_iters)[0]

    return torch.argmax(gamma, dim=-1)
