"""Fixed 280x280 Depth Anything V2-Small adapter for DS/U250 export."""

from __future__ import annotations

import torch

from depth_anything_v2.dpt import DepthAnythingV2
from ds_models.depth_anything_v2_vits import (
    CHECKPOINT,
    MODEL_CONFIG,
    _install_ds_attention,
    _install_ds_deconvolutions,
    _install_runtime_cls_token,
    load_checkpoint,
)


ifmap_sz = (3, 280, 280)
input_layouts = ("CHW",)
input_names = ("input0",)
op_version = 18
batch_size = 1
batch_onnx = False


class Model(DepthAnythingV2):
    """The existing ViT-S weights with a fixed 20x20 patch-grid contract."""

    def __init__(self) -> None:
        super().__init__(**MODEL_CONFIG)
        load_checkpoint(self, CHECKPOINT)
        _install_runtime_cls_token(self)
        _install_ds_attention(self)
        _install_ds_deconvolutions(self)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != (1, 3, 280, 280):
            raise ValueError(
                "Depth Anything V2 280 DS adapter requires input shape "
                f"[1, 3, 280, 280], got {tuple(x.shape)}"
            )
        return super().forward(x)
