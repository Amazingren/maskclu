from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.svm import SVC

from modules.ema import EMA
from modules.networks import PointTokenizer, PointMasking, PosE
from modules.transformer import TransformerEncoder, TransformerEncoderOutput


class Point2Vec(nn.Module):
    def __init__(
            self,
            tokenizer_num_groups: int = 64,
            tokenizer_group_size: int = 32,
            d2v_masking_ratio: float = 0.65,
            d2v_masking_type: str = "rand",  # rand, block
            encoder_dim: int = 384,
            encoder_depth: int = 12,
            encoder_heads: int = 6,
            encoder_dropout: float = 0,
            encoder_attention_dropout: float = 0.05,
            encoder_drop_path_rate: float = 0.25,
            encoder_add_pos_at_every_layer: bool = True,
            decoder: bool = True,
            decoder_depth: int = 4,
            decoder_dropout: float = 0,
            decoder_attention_dropout: float = 0.05,
            decoder_drop_path_rate: float = 0.25,
            decoder_add_pos_at_every_layer: bool = True,
            loss: str = "smooth_l1",  # smooth_l1, mse
            train_transformations: List[str] = [
                "subsample",
                "scale",
                "center",
                "unit_sphere",
                "rotate",
            ],
            val_transformations: List[str] = ["subsample", "center", "unit_sphere"],
    ) -> None:
        super(Point2Vec, self).__init__()

        self.train_transformations = self.build_transformations(train_transformations)
        self.val_transformations = self.build_transformations(val_transformations)

        self.positional_encoding = nn.Sequential(
            PosE(3, 6),
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )

        self.tokenizer = PointTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            token_dim=encoder_dim,
        )

        self.masking = PointMasking(ratio=d2v_masking_ratio, type=d2v_masking_type)

        init_std = 0.02
        self.mask_token = nn.Parameter(torch.zeros(encoder_dim))
        nn.init.trunc_normal_(self.mask_token, mean=0, std=init_std, a=-init_std, b=init_std)

        dpr = [x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)]
        decoder_dpr = [x.item() for x in torch.linspace(0, decoder_drop_path_rate, decoder_depth)]

        self.student = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        if decoder:
            self.decoder = TransformerEncoder(
                embed_dim=encoder_dim,
                depth=decoder_depth,
                num_heads=encoder_heads,
                qkv_bias=True,
                drop_rate=decoder_dropout,
                attn_drop_rate=decoder_attention_dropout,
                drop_path_rate=decoder_dpr,
                add_pos_at_every_layer=decoder_add_pos_at_every_layer,
            )
            self.regressor = nn.Linear(encoder_dim, encoder_dim)
        else:
            self.regressor = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.GELU(),
                nn.Linear(encoder_dim, encoder_dim),
            )

        if loss == "mse":
            self.loss_func = nn.MSELoss()
        elif loss == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(beta=2)
        else:
            raise ValueError(f"Unknown loss: {loss}")

    def build_transformations(self, transformations: List[str]):
        # Implement your transformation building logic here
        pass

    def setup(self) -> None:
        self.teacher = EMA(
            self.student,
            tau_min=0
            if self.hparams.d2v_ema_tau_min is None  # type: ignore
            else self.hparams.d2v_ema_tau_min,  # type: ignore
            tau_max=1
            if self.hparams.d2v_ema_tau_max is None  # type: ignore
            else self.hparams.d2v_ema_tau_max,  # type: ignore
            tau_steps=(self.hparams.fix_estimated_stepping_batches or self.trainer.estimated_stepping_batches
                       ) * (self.hparams.d2v_ema_tau_epochs / self.trainer.max_epochs),  # type: ignore
            update_after_step=0,
            update_every=1,
        )

    def forward(self, embeddings: torch.Tensor, centers: torch.Tensor, mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        w = mask.unsqueeze(-1).type_as(embeddings)
        corrupted_embeddings = (1 - w) * embeddings + w * self.mask_token

        if hasattr(self, 'decoder'):
            B, _, C = embeddings.shape
            visible_embeddings = corrupted_embeddings[~mask].reshape(B, -1, C)
            masked_embeddings = corrupted_embeddings[mask].reshape(B, -1, C)

            pos = self.positional_encoding(centers)
            visible_pos = pos[~mask].reshape(B, -1, C)
            masked_pos = pos[mask].reshape(B, -1, C)
            output_embeddings = self.student(visible_embeddings, visible_pos).last_hidden_state

            decoder_output_tokens = self.decoder(
                torch.cat([output_embeddings, masked_embeddings], dim=1),
                torch.cat([visible_pos, masked_pos], dim=1),
            ).last_hidden_state

            predictions = self.regressor(
                decoder_output_tokens[:, -masked_embeddings.shape[1]:].reshape(-1, C)
            )
        else:
            pos = self.positional_encoding(centers)
            output_embeddings = self.student(corrupted_embeddings, pos).last_hidden_state
            predictions = self.regressor(output_embeddings[mask])

        targets = self.generate_targets(embeddings, pos)[mask]
        return predictions, targets

    def generate_targets(self, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Implement your logic to generate targets
        assert self.teacher.ema_model is not None  # always false
        self.teacher.ema_model.eval()
        d2v_target_layers: List[int] = self.hparams.d2v_target_layers  # type: ignore
        d2v_target_layer_part: str = self.hparams.d2v_target_layer_part  # type: ignore
        output: TransformerEncoderOutput = self.teacher(
            tokens,
            pos,
            return_hidden_states=d2v_target_layer_part == "final",
            return_ffns=d2v_target_layer_part == "ffn",
        )
        if d2v_target_layer_part == "ffn":
            assert output.ffns is not None
            target_layers = output.ffns
        elif d2v_target_layer_part == "final":
            assert output.hidden_states is not None
            target_layers = output.hidden_states
        else:
            raise ValueError()
        target_layers = [
            target_layers[i] for i in d2v_target_layers
        ]  # [(B, T, C)]
        # pre norm
        target_layer_norm = self.hparams.d2v_target_layer_norm  # type: ignore
        if target_layer_norm == "instance":
            target_layers = [
                F.instance_norm(target.transpose(1, 2)).transpose(1, 2)
                for target in target_layers
            ]
        elif target_layer_norm == "layer":
            target_layers = [
                F.layer_norm(target, target.shape[-1:]) for target in target_layers
            ]
        elif target_layer_norm == "group":
            target_layers = [
                F.layer_norm(target, target.shape[-2:]) for target in target_layers
            ]
        elif target_layer_norm == "batch":
            target_layers = [
                F.batch_norm(
                    target.transpose(1, 2),
                    running_mean=None,
                    running_var=None,
                    training=True,
                ).transpose(1, 2)
                for target in target_layers
            ]
        elif target_layer_norm is not None:
            raise ValueError()

        # Average top K blocks
        targets = torch.stack(target_layers, dim=0).mean(0)  # (B, T, C)

        # post norm

        target_norm = self.hparams.d2v_target_norm  # type: ignore
        if target_norm == "instance":
            targets = F.instance_norm(targets.transpose(1, 2)).transpose(1, 2)
        elif target_norm == "layer":
            targets = F.layer_norm(targets, targets.shape[-1:])
        elif target_norm == "group":
            targets = F.layer_norm(targets, targets.shape[-2:])
        elif target_norm == "batch":
            targets = F.batch_norm(
                targets.transpose(1, 2),
                running_mean=None,
                running_var=None,
                training=True,
            ).transpose(1, 2)
        elif target_norm is not None:
            raise ValueError()

        return targets

    def training_step(self, points: torch.Tensor):
        points = self.train_transformations(points)
        x, y = self._perform_step(points)
        loss = self.loss_func(x, y)
        return loss

    def validation_step(self, points: torch.Tensor):
        points = self.val_transformations(points)
        x, y = self._perform_step(points)
        loss = self.loss_func(x, y)
        return loss

    def _perform_step(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, centers = self.tokenizer(inputs)
        mask = self.masking(centers)
        return self.forward(tokens, centers, mask)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer


# Main training loop
def train_model(model, train_loader, val_loader, num_epochs=100):
    optimizer = model.configure_optimizers()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            points = batch[0]  # Assuming batch contains (points, labels)
            optimizer.zero_grad()
            loss = model.training_step(points)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                points = batch[0]
                loss = model.validation_step(points)
                val_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader)}")

# Usage:
# model = Point2Vec()
# train_model(model, train_loader, val_loader, num_epochs=100)
