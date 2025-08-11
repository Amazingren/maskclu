import torch
import torch.nn.functional as F
from fvcore.common.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from timm.layers import trunc_normal_
from torch import nn

from libs.lib_utils import index_points
from modules.networks import Encoder, Group, get_pos_embed
from modules.transformer import Block


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = torch.cdist(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class SegTransformer(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 6

        self.group_size = 32
        self.num_group = 128
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        # self.pos_embed = nn.Sequential(
        #     nn.Linear(3, 128),
        #     nn.GELU(),
        #     nn.Linear(128, self.trans_dim)
        # )
        self.pos_embed = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.label_conv_cls = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(64),
        #                            nn.LeakyReLU(0.2))

        self.propagation_0_cls = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024])

        self.convs1_cls = nn.Conv1d(3328, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2_cls = nn.Conv1d(512, 256, 1)
        self.convs3_cls = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1_cls = nn.BatchNorm1d(512)
        self.bns2_cls = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    @staticmethod
    def get_loss(pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight)
        return total_loss

    def load_model_from_ckpt(self, bert_ckpt_path, model_key='MAE_encoder'):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith(model_key):
                    base_ckpt[k[len(model_key + '.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)

                )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def load_model_from_ckpt_withrename(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)['model_state_dict']
            model_dict = self.state_dict()
            for k in list(model_dict.keys()):
                if k in ckpt:
                    model_dict[k] = ckpt[k]
                else:
                    old_k = k.replace("_cls", "")
                    print(old_k, k)
                    model_dict[k] = ckpt[old_k]
            # base_ckpt = {k.replace("_cls.", ""): v for k, v in ckpt['model_state_dict'].items()}
            # print(ckpt['base_model'].items())
            # for k in list(base_ckpt.keys()):
            #     if k.startswith('MAE_encoder'):
            #         base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
            #         del base_ckpt[k]
            #     elif k.startswith('base_model'):
            #         base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            #         del base_ckpt[k]

            incompatible = self.load_state_dict(model_dict, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)

                )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts):
        B, C, N = pts.shape
        # print(pts.shape)  ## 16 * 3 * 2048
        pts = pts.transpose(-1, -2)  # B N 3
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)  # B G N

        pos = self.pos_embed(get_pos_embed(self.trans_dim, center))
        # final input
        x = group_input_tokens
        # transformer
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # 1152
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        # cls_label_one_hot = cls_label.view(B, 16, 1)
        # cls_label_feature = self.label_conv_cls(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1152*2

        f_level_0 = self.propagation_0_cls(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)

        x = torch.cat((f_level_0, x_global_feature), 1)
        x = self.relu(self.bns1_cls(self.convs1_cls(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2_cls(self.convs2_cls(x)))
        x = self.convs3_cls(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x
