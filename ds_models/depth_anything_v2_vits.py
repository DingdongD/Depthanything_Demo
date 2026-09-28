"""Fixed-shape Depth Anything V2-Small adapter for the DS toolchain.

The DS exporter expects a module exposing ``Model`` and input metadata.  This
adapter deliberately keeps preprocessing outside the compiled graph: the
compiled model consumes a normalized RGB NCHW tensor of shape
``[1, 3, 518, 518]`` and returns ``[1, 518, 518]``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn

from depth_anything_v2.dpt import DepthAnythingV2
from ds_models.static_int8_attention import (
    HardwareRunner,
    StaticAttentionProfile,
    StaticInt8AttentionRuntime,
    plan_query_chunks,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = os.environ.get(
    "DEPTH_ANYTHING_CHECKPOINT",
    str(ROOT / "checkpoints" / "depth_anything_v2_vits.pth"),
)

ifmap_sz = (3, 518, 518)
input_layouts = ("CHW",)
input_names = ("input0",)
op_version = 18
batch_size = 1
batch_onnx = False

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


class DSAttention4D(nn.Module):
    """Attention implementation whose exported intermediates are rank <= 4.

    The reference DINOv2 implementation reshapes QKV to ``[B, N, 3, H, D]``
    and permutes it, which introduces a rank-5 tensor in ONNX.  ACMOSA/DS
    physical layouts currently support at most rank 4.  Splitting the linear
    output before the head reshape is mathematically identical and keeps each
    Q/K/V tensor at rank 4: ``[B, N, C] -> [B, H, N, D]``.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False,
                 proj_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, query_chunk_rows: int = 256,
                 layer_index: int | None = None) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"attention dimension {dim} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query_chunk_rows = query_chunk_rows
        self.layer_index = layer_index
        self.static_runtime: StaticInt8AttentionRuntime | None = None
        self.static_export_profile: StaticAttentionProfile | None = None
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q * self.scale
        if self.static_runtime is not None:
            if self.training:
                raise RuntimeError("static U250 attention is inference-only")
            if self.layer_index is None:
                raise RuntimeError("static U250 attention requires a layer index")
            out_heads = self.static_runtime.execute(self.layer_index, q, k, v)
        elif self.static_export_profile is not None:
            if self.layer_index is None:
                raise RuntimeError("static U250 export requires a layer index")
            if n != self.static_export_profile.tokens:
                raise ValueError(
                    f"static U250 export expects {self.static_export_profile.tokens} "
                    f"tokens, got {n}"
                )
            # Keep each head and query chunk as an independent MatMul/Softmax/
            # MatMul triplet in ONNX. A post-export pass attaches the calibrated
            # static-INT8 attributes to these nodes. Multiplying V by the
            # calibrated gain before AV lets the AV B scale be
            # v_scale*gain while preserving round(V/v_scale) integer values.
            per_head = []
            for head in range(self.num_heads):
                scale = self.static_export_profile.scale(self.layer_index, head)
                query = q[:, head:head + 1]
                key_transposed = k[:, head:head + 1].transpose(-2, -1)
                value = v[:, head:head + 1] * scale.av_output_gain
                outputs = []
                for chunk in plan_query_chunks(n, self.query_chunk_rows):
                    # The standalone runtime pads the tail because it reuses
                    # one 256-row kernel.  A unified model has six distinct
                    # instructions, so keep the final logical M=90.  ACPC
                    # supplies its normal physical alignment and no constant
                    # zero tensor needs to cross the FDDR path.
                    query_slice = query[:, :, chunk.start:chunk.stop]
                    probability = torch.matmul(
                        query_slice, key_transposed
                    ).softmax(dim=-1)
                    output = torch.matmul(probability, value)
                    outputs.append(output)
                # Collapse the statically singleton head axis before joining
                # heads.  A rank-4 NDWC concat on its channel axis is rejected
                # by the legacy ACPC liveness mapper after inplace-root
                # canonicalization; the equivalent rank-3 feature concat is
                # a supported path and already has the desired [B,N,C] order.
                per_head.append(
                    torch.cat(outputs, dim=-2).reshape(b, n, self.head_dim)
                )
            out = torch.cat(per_head, dim=-1)
            return self.proj_drop(self.proj(out))
        else:
            # Keep every CTC MatMul left/query dimension <=256. K and V remain
            # global, so concatenating the six query slices is exactly
            # equivalent to full 1370x1370 self-attention in eval mode.
            key_transposed = k.transpose(-2, -1)
            chunks = []
            for chunk in plan_query_chunks(n, self.query_chunk_rows):
                probability = torch.matmul(
                    q[:, :, chunk.start:chunk.stop], key_transposed
                ).softmax(dim=-1)
                chunks.append(torch.matmul(self.attn_drop(probability), v))
            out_heads = torch.cat(chunks, dim=-2)
        out = out_heads.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class DSConvTranspose2d(nn.Module):
    """Exact no-overlap ConvTranspose replacement for DS export.

    The two Depth Anything decoder deconvolutions use ``kernel_size ==
    stride`` and zero padding, so every input pixel contributes to one output
    tile without overlap.  Packing the tile coefficients into a 1x1 Conv2d
    followed by PixelShuffle is numerically identical while lowering to the
    DS-supported Conv/DepthToSpace operators.
    """

    def __init__(self, source: nn.ConvTranspose2d) -> None:
        super().__init__()
        stride = source.stride
        kernel = source.kernel_size
        if (source.padding != (0, 0) or source.output_padding != (0, 0)
                or source.dilation != (1, 1) or source.groups != 1
                or stride[0] != stride[1] or kernel != stride):
            raise ValueError("decoder ConvTranspose is not an exact phase-expand case")
        r = stride[0]
        self.conv = nn.Conv2d(
            source.in_channels,
            source.out_channels * r * r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=source.bias is not None,
        )
        with torch.no_grad():
            # ConvTranspose2d weights are [Cin, Cout, Kh, Kw]; PixelShuffle
            # consumes [Cout, Kh, Kw] tiles in row-major phase order.
            packed = torch.empty(
                source.out_channels * r * r, source.in_channels, 1, 1,
                dtype=source.weight.dtype, device=source.weight.device,
            )
            if r == 4:
                # DS supports PixelShuffle block size 2 only.  Two shuffles
                # are equivalent to block size 4; arrange channels so the
                # first shuffle supplies the low phase and the second the
                # high phase.
                for out_ch in range(source.out_channels):
                    for high_h in range(2):
                        for high_w in range(2):
                            intermediate = out_ch * 4 + high_h * 2 + high_w
                            for low_h in range(2):
                                for low_w in range(2):
                                    packed_idx = intermediate * 4 + low_h * 2 + low_w
                                    src_h = 2 * low_h + high_h
                                    src_w = 2 * low_w + high_w
                                    packed[packed_idx, :, 0, 0] = source.weight[:, out_ch, src_h, src_w]
            else:
                packed.copy_(source.weight.permute(1, 2, 3, 0).reshape_as(packed))
            self.conv.weight.copy_(packed)
            if source.bias is not None:
                self.conv.bias.copy_(source.bias.repeat_interleave(r * r))
        if r == 4:
            self.shuffle_low = nn.PixelShuffle(2)
            self.shuffle_high = nn.PixelShuffle(2)
        else:
            self.shuffle = nn.PixelShuffle(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.conv(x)
        if hasattr(self, "shuffle_low"):
            return self.shuffle_high(self.shuffle_low(expanded))
        return self.shuffle(expanded)


def _install_ds_attention(model: nn.Module) -> None:
    """Replace every DINO attention block with the rank-4 export variant."""

    layer_index = 0
    for module in model.modules():
        if not hasattr(module, "attn"):
            continue
        old = module.attn
        if not hasattr(old, "qkv") or not hasattr(old, "proj"):
            continue
        new = DSAttention4D(
            old.qkv.in_features,
            old.num_heads,
            qkv_bias=old.qkv.bias is not None,
            proj_bias=old.proj.bias is not None,
            attn_drop=old.attn_drop.p,
            proj_drop=old.proj_drop.p,
            layer_index=layer_index,
        )
        new.qkv.load_state_dict(old.qkv.state_dict())
        new.proj.load_state_dict(old.proj.state_dict())
        new.eval()
        module.attn = new
        layer_index += 1


def _install_runtime_cls_token(model: nn.Module) -> None:
    """Give the class token a normal activation producer in the DS graph.

    The ONNX exporter otherwise wires ``cls_token`` directly into Concat.  ACPC
    treats Concat operands as activation buffers, while an initializer has no
    producer/liveness interval.  Anchoring the token to a dynamic patch token
    with an exact zero multiply preserves the model result and makes the Add
    output an ordinary runtime activation.
    """

    backbone = model.pretrained

    def prepare_tokens_with_masks_ds(self, x, masks=None):
        _, _, width, height = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.to(x.dtype).unsqueeze(0),
                x,
            )

        runtime_cls_token = x[:, :1, :] * 0.0 + self.cls_token.to(x.dtype)
        x = torch.cat((runtime_cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, width, height)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    backbone.prepare_tokens_with_masks = MethodType(
        prepare_tokens_with_masks_ds, backbone
    )


def _install_ds_deconvolutions(model: nn.Module) -> None:
    """Replace decoder ConvTranspose layers with exact phase-expand blocks."""

    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.ConvTranspose2d):
                setattr(parent, name, DSConvTranspose2d(child))


def load_checkpoint(model: torch.nn.Module, checkpoint: str | os.PathLike[str] = CHECKPOINT) -> torch.nn.Module:
    """Load the official checkpoint with strict key checking."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Depth Anything checkpoint not found: {checkpoint_path}. "
            "Download depth_anything_v2_vits.pth into checkpoints/."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


class Model(DepthAnythingV2):
    """Depth Anything V2-Small with a DS-compatible fixed input contract."""

    def __init__(self) -> None:
        super().__init__(**MODEL_CONFIG)
        load_checkpoint(self)
        _install_runtime_cls_token(self)
        _install_ds_attention(self)
        _install_ds_deconvolutions(self)
        self.eval()

    def enable_static_int8_attention(
        self,
        profile_path: str | os.PathLike[str],
        runner: HardwareRunner | None = None,
    ) -> "Model":
        """Route all 12 attention layers through calibrated six-slice runtime.

        A production U250 runner uses ``layer`` and ``head`` from the callback
        to select the corresponding entry in ``runtime_manifest.json``. Passing
        no runner enables the numerically equivalent static-INT8 surrogate.
        """

        runtime = StaticInt8AttentionRuntime(
            StaticAttentionProfile.load(profile_path), runner=runner
        )
        attention_layers = [m for m in self.modules() if isinstance(m, DSAttention4D)]
        if len(attention_layers) != len(runtime.profile.layers):
            raise ValueError(
                f"model has {len(attention_layers)} attention layers but profile has "
                f"{len(runtime.profile.layers)}"
            )
        for layer_index, attention in enumerate(attention_layers):
            attention.layer_index = layer_index
            attention.static_runtime = runtime
        return self

    def enable_static_int8_attention_export(
        self, profile_path: str | os.PathLike[str]
    ) -> "Model":
        """Expose 432 calibrated attention triplets in the exported ONNX graph."""

        profile = StaticAttentionProfile.load(profile_path)
        attention_layers = [m for m in self.modules() if isinstance(m, DSAttention4D)]
        if len(attention_layers) != len(profile.layers):
            raise ValueError(
                f"model has {len(attention_layers)} attention layers but profile has "
                f"{len(profile.layers)}"
            )
        for layer_index, attention in enumerate(attention_layers):
            attention.layer_index = layer_index
            attention.static_runtime = None
            attention.static_export_profile = profile
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != (1, 3, 518, 518):
            raise ValueError(
                "Depth Anything V2 DS adapter requires input shape "
                f"[1, 3, 518, 518], got {tuple(x.shape)}"
            )
        return super().forward(x)
