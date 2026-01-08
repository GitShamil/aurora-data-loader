import contextlib
import dataclasses
import warnings
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from src.dataloader.batch import Batch

from src.model.decoder import Perceiver3DDecoder
from src.model.encoder import Perceiver3DEncoder

__all__ = [
    "AuroraAE",
]


class AuroraAE(torch.nn.Module):

    default_checkpoint_repo = "microsoft/aurora"
    """str: Name of the HuggingFace repository to load the default checkpoint from."""

    default_checkpoint_name = "aurora-0.25-finetuned.ckpt"
    """str: Name of the default checkpoint."""

    default_checkpoint_revision = "0be7e57c685dac86b78c4a19a3ab149d13c6a3dd"
    """str: Commit hash of the default checkpoint."""

    def __init__(
        self,
        *,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        stabilise_level_agg: bool = False,
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        level_condition: Optional[tuple[int | float, ...]] = None,
        dynamic_vars: bool = False,
        atmos_static_vars: bool = False,
        separate_perceiver: tuple[str, ...] = (),
        positive_surf_vars: tuple[str, ...] = (),
        positive_atmos_vars: tuple[str, ...] = (),
    ) -> None:
        """Construct an instance of the model.

        Args:
            surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model.
            static_vars (tuple[str, ...], optional): All static variables supported by the
                model.
            atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model.
            latent_levels (int, optional): Number of latent pressure levels.
            patch_size (int, optional): Patch size.
            embed_dim (int, optional): Patch embedding dimension.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks. The dimensionality of these attention heads will be equal to
                `embed_dim` divided by this value.
            mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs.
            drop_rate (float, optional): Drop-out rate.
            enc_depth (int, optional): Number of Perceiver blocks in the encoder.
            dec_depth (int, optional): Number of Perceiver blocks in the decoder.
            dec_mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs in the
                decoder. The embedding dimensionality here is different, which is why this is a
                separate parameter.
            perceiver_ln_eps (float, optional): Epsilon in the perceiver layer norm. layers. Used
                to stabilise the model.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
            surf_stats (dict[str, tuple[float, float]], optional): For these surface-level
                variables, adjust the normalisation to the given tuple consisting of a new location
                and scale.
            level_condition (tuple[int | float, ...], optional): Make the patch embeddings dependent
                on pressure level. If you want to enable this feature, provide a tuple of all
                possible pressure levels.
            dynamic_vars (bool, optional): Use dynamically generated static variables, like time
                of day. Defaults to `False`.
            atmos_static_vars (bool, optional): Also concatenate the static variables to the
                atmospheric variables. Defaults to `False`.
            separate_perceiver (tuple[str, ...], optional): In the decoder, use a separate Perceiver
                for specific atmospheric variables. This can be helpful at fine-tuning time to deal
                with variables that have a significantly different behaviour. If you want to enable
                this features, set this to the collection of variables that should be run on a
                separate Perceiver.
            positive_surf_vars (tuple[str, ...], optional): Mark these surface-level variables as
                positive. Clamp them before running them through the encoder, and also clamp them
                when autoregressively rolling out the model. The variables are not clamped for the
                first roll-out step.
            positive_atmos_vars (tuple[str, ...], optional): Mark these atmospheric variables as
                positive. Clamp them before running them through the encoder, and also clamp them
                when autoregressively rolling out the model. The variables are not clamped for the
                first roll-out step.
        """
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or dict()
        self.positive_surf_vars = positive_surf_vars
        self.positive_atmos_vars = positive_atmos_vars

        if self.surf_stats:
            warnings.warn(
                f"The normalisation statics for the following surface-level variables are manually "
                f"adjusted: {', '.join(sorted(self.surf_stats.keys()))}. "
                f"Please ensure that this is right!",
                stacklevel=2,
            )

        self.encoder = Perceiver3DEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=embed_dim // num_heads,
            depth=enc_depth,
            latent_levels=latent_levels,
            perceiver_ln_eps=perceiver_ln_eps,
            stabilise_level_agg=stabilise_level_agg,
            level_condition=level_condition,
            dynamic_vars=dynamic_vars,
            atmos_static_vars=atmos_static_vars,
        )

        self.latent_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2, eps=perceiver_ln_eps)
        )

        self.decoder = Perceiver3DDecoder(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            # Concatenation at the backbone end doubles the dim.
            embed_dim=embed_dim * 2,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            depth=dec_depth,
            # Because of the concatenation, high ratios are expensive.
            # We use a lower ratio here to keep the memory in check.
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
            level_condition=level_condition,
            separate_perceiver=separate_perceiver,
#            modulation_heads=modulation_heads,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`aurora.Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        batch = self.batch_transform_hook(batch)

        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        # Apply some transformations before feeding `batch` to the encoder. We'll later want to
        # refer to the original batch too, so rename the variable.
        transformed_batch = batch

        transformed_batch = self._pre_encoder_hook(transformed_batch)

        # The encoder is always just run.
        x = self.encoder(
            transformed_batch,
        )

        x = self.latent_projection(x)

        pred = self.decoder(
            x,
            batch,
            patch_res=patch_res,
        )

        # Remove batch and history dimension from static variables.
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction. The time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        pred = self._post_decoder_hook(batch, pred)

        if self.positive_surf_vars:
            pred = dataclasses.replace(
                pred,
                surf_vars={
                    k: v.clamp(min=0) if k in self.positive_surf_vars else v
                    for k, v in pred.surf_vars.items()
                },
            )
        if self.positive_atmos_vars:
            pred = dataclasses.replace(
                pred,
                atmos_vars={
                    k: v.clamp(min=0) if k in self.positive_atmos_vars else v
                    for k, v in pred.atmos_vars.items()
                },
            )

        pred = pred.unnormalise(surf_stats=self.surf_stats)

        return pred

    def batch_transform_hook(self, batch: Batch) -> Batch:
        """Transform the batch right after receiving it and before normalisation.

        This function should be idempotent.
        """
        return batch

    def _pre_encoder_hook(self, batch: Batch) -> Batch:
        """Transform the batch before it goes through the encoder."""
        return batch

    def _post_decoder_hook(self, batch: Batch, pred: Batch) -> Batch:
        """Transform the prediction right after the decoder."""
        return pred