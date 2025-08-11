import copy

import torch
from torch import nn

from libs.knn_graph import label_to_centers, knn_based_affinity_matrix, normalized_laplacian
from libs.loss import loss_fn, info_nce_loss, contrastive_loss_2d, welsch_loss, spatial_loss, contrastive_loss_3d
from libs.mincut import self_balanced_min_cut_batch, spectrum_clustering, wkmeans
from modules.networks import PosE, Group, MinCutPoolingLayer
from modules.transformer import MaskTransformer
from modules.transformer import TransformerDecoder
from tools.logger import print_log


def _momentum_update_key_encoder(q_encoder, k_encoder, m):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(copy.deepcopy(q_encoder).parameters(), k_encoder.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

    return k_encoder


class MaskClu(nn.Module):
    def __init__(self, config, device='auto'):
        super().__init__()
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # print("Using device: {}".format(device))
            self.device = torch.device(device)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise NotImplementedError
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.n_neighs = config.n_neighs
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_ratio = config.transformer_config.mask_ratio
        self.proj_dim = config.transformer_config.proj_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.MAE_encoder = MaskTransformer(config)

        self.decoder_pos_embed = nn.Sequential(
            PosE(3, 48),
            nn.Linear(48, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.n_clusters = config.transformer_config.decoder_num_clusters
        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.norm = nn.LayerNorm(self.trans_dim)

        # prediction head
        self.cut_pool = MinCutPoolingLayer(self.trans_dim, self.proj_dim, self.n_clusters, config.tau)
        # self.protos = torch.randn(self.n_clusters, self.proj_dim, requires_grad=True, device=self.device)
        self.to(self.device)

    def clustering(self, xyz, feats, mask=None, is_cus='balance', is_log=False):
        adj = knn_based_affinity_matrix(xyz, feats, k=self.n_neighs, gamma=-1, is_zeros=True)
        n_neighs = (adj > 0).float().sum(dim=-1).mean()
        if n_neighs < self.n_neighs / 4:
            print('[Point_MAE] clustering need more neighbors, current neighboors are {}'.format(n_neighs))
        s = None
        if is_cus == 'balance':
            assign = self_balanced_min_cut_batch(adj, self.n_clusters, feats=xyz)
            c_fea = label_to_centers(feats, assign, is_norm=True)
        elif is_cus == 'kmeans':
            assign = wkmeans(feats, self.n_clusters, dst='feats', iters=10, in_iters=5)
            c_fea = label_to_centers(feats, assign, is_norm=True)
        else:
            _, norm_adj = normalized_laplacian(adj, 1e-4)
            c_fea, assign, s = self.cut_pool(feats, norm_adj.detach())
        c_xyz = label_to_centers(xyz, assign, is_norm=True)
        if mask is not None:
            B = assign.size(0)
            mask_assign = assign[mask].reshape(B, -1, self.n_clusters)
            unmask_assign = assign[~mask].reshape(B, -1, self.n_clusters)
            assign = torch.cat([mask_assign, unmask_assign], dim=1)
        if is_log:
            return c_xyz, c_fea, s

        return c_xyz, c_fea, assign

    def extractor(self, neighborhood, center, is_new=False):
        cls_fea, x_vis, mask = self.MAE_encoder(neighborhood, center, is_new=is_new)
        B, M, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask].reshape(B, -1, 3))
        pos_emd_mask = self.decoder_pos_embed(center[mask].reshape(B, -1, 3))
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        x_rec = self.norm(x_rec)
        # B M 1024
        mask_cpts = center[mask].reshape(B, -1, 3)
        unmask_cpts = center[~mask].reshape(B, -1, 3)
        cts_shift = torch.cat([mask_cpts, unmask_cpts], dim=1)
        x = torch.cat([x_rec, x_vis], dim=1)
        return cls_fea, cts_shift, x, x_rec, mask

    def forward(self, pts, is_eval=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        cls_fea, patch_fea = self.MAE_encoder(neighborhood, center, is_eval=True)
        if is_eval:
            return patch_fea.max(dim=1)[0]
        q_cls, cts_shift, q_fea, x_rec, mask = self.extractor(neighborhood, center)
        # cls_token_k, x_rec_k, c_xyz_k, loss_k, x_mask_k = self.extractor(neighborhood, center, is_new=True)
        # l_loss = spatial_loss(cts_shift, assign, tau=1.0)
        B, _, emd_dim = patch_fea.size()
        # mask_k_fea = patch_fea[mask].reshape(-1, emd_dim)
        # g_loss = contrastive_loss_3d(x_rec, mask_k_fea.detach().reshape(B, -1, emd_dim), tau=0.1, is_norm=True)
        g_loss = loss_fn(q_cls, patch_fea.max(dim=1)[0].detach()) + loss_fn(
            q_cls.detach(), x_rec.max(dim=1)[0])
        # c_k_xyz, c_k_fea, k_assign = self.clustering(center, patch_fea, mask, is_cus='mincut')
        c_q_xyz, c_q_fea, q_log_score = self.clustering(cts_shift, q_fea, is_cus='mincut', is_log=True)
        assign_loss = spatial_loss(cts_shift, c_q_xyz, q_log_score, tau=0.5)
        rec_loss = 0.1*welsch_loss(center, c_q_xyz, sigma=1.0)

        return rec_loss + g_loss, assign_loss
