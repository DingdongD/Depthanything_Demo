# Depth Anything V2 518x518 on U250: workflow, tensor shapes, and workload allocation

## 1. Scope and source of truth

This document describes the current production 518x518 path used by the App:

- Runtime profile: `r200 518 unified final-depth multi-domain SOTA`
- U250 package: generated with `tools/package_u250_base.py` (historical board
  qualification label: `depthanything_u250_sota_518`)
- Runtime contract: `depthanything_u250_runtime_contract.json`
- Model input: `[1, 3, 518, 518]`
- Model output: `[1, 518, 518]` relative inverse depth
- Encoder: ViT-S, 12 blocks, hidden dimension 384, 6 attention heads
- Decoder: DPT reassemble + scratch projections + RefineNet4/3/2/1 + depth head

The tensor shapes in this document are logical model shapes. U250 uses packed,
banked physical layouts internally, and some logical operators are divided into
channel or row tiles.

### Backend legend

| Label | Meaning |
|---|---|
| App CPU | Local App process: camera input, image preprocessing, transfer, metric alignment, and visualization |
| Host C++ | x86 host attached to U250: native host graph, layout codec, quantization, elementwise and shape operations |
| NPU | U250 accelerator: convolution, linear/MatMul, LayerNorm core, SPU softmax, and attention kernels |

The hot host tensor path uses the native C++ executor. Python remains the
resident-process controller and grouped-dispatch orchestrator; it is not used
for per-element NumPy execution in the qualified path.

## 2. End-to-end data flow

```text
Nebula411 RGB [600,800,3]
  |
  | App CPU: resize + ImageNet normalization + HWC-to-NCHW
  v
Model input [1,3,518,518] FP32
  |
  | Host C++: exact 14x14 patchify / pixel rearrangement
  v
[1,588,37,37]
  |
  | NPU: patch projection, 6 output-channel kernels
  v
[1,384,37,37] BF16
  |
  | Host C++: transpose + reshape + CLS concat + position add
  v
Tokens [1,1370,384]
  |
  | 12 x Encoder Block
  |   NPU: Norm core, QKV, attention, post projection, FC1, FC2
  |   Host: layout/pack, GELU, residual Add
  v
Block outputs [1,1370,384]
  |
  +-- capture Block2  --+
  +-- capture Block5  --+--> DPT four-level reassemble
  +-- capture Block8  --+
  +-- capture Block11 --+
  |
  | Host: final norms and token-to-feature conversion
  | NPU: project/resize/scratch convolutions
  v
L1 [1,64,148,148]
L2 [1,64, 74, 74]
L3 [1,64, 37, 37]
L4 [1,64, 19, 19]
  |
  | RefineNet4 -> RefineNet3 -> RefineNet2 -> RefineNet1
  | NPU: all Conv
  | Host: ReLU, shortcut Add, cross-scale Add, Resize
  v
[1,64,296,296]
  |
  | NPU/Host depth output head
  v
Raw relative inverse depth [1,518,518]
  |
  | App CPU: ToF-guided inverse-depth scale/shift alignment
  v
Metric depth [600,800], metres
```

Depth Anything is RGB-only. Nebula411 ToF does not enter the neural network;
it is used after inference for metric scale/shift alignment, evaluation, and
point-cloud reconstruction.

## 3. Runtime-level allocation summary

| Stage | Main NPU workload | Main Host workload | Logical NPU calls |
|---|---|---|---:|
| Frontend | Patch projection | Patchify and token assembly | 6 |
| Encoder, 12 blocks | Norm core, QKV, QK, softmax, dual AV, post projection, FC1, FC2 | Pack/layout, GELU, residual Add, Block0 Head3 attention | 312 |
| Decoder | 32 source Conv modules | Final LN, ReLU, Add, Resize, DepthToSpace and layout | 89 |
| Total | Compute-intensive kernels | Control, layout and non-linear glue | 407 |

The 407 entries are logical kernel calls in the runtime contract. C++ frame
graphs and grouped launches reduce these to about 239 physical dispatches in
the current measured path.

Resident state:

| Item | Value |
|---|---:|
| Resident bank size | 51,326,976 bytes |
| Resident kernels | 237 |
| Weight-bank load | Once per resident process |
| Static H2C writes | 2 per resident process |

## 4. Frontend and patch embedding

| Order | Module | Backend | Input | Output | Notes |
|---:|---|---|---|---|---|
| 1 | RGB resize | App CPU | `[600,800,3]` | `[518,518,3]` | Bilinear |
| 2 | ImageNet normalization | App CPU | `[518,518,3]` | `[1,3,518,518]` | FP32 NCHW |
| 3 | Patchify | Host C++ | `[1,3,518,518]` | `[1,588,37,37]` | Exact 14x14 pixel rearrangement |
| 4 | Input quantize/pack | Host C++ | `[1,588,37,37]` | Same logical shape, INT8 | Symmetric scale about 0.0307386 |
| 5 | Patch projection | NPU | `[1,588,37,37]` | `[1,384,37,37]` | A8 x B8, BF16 output; 6 x 64-channel kernels |
| 6 | Transpose + reshape | Host C++ | `[1,384,37,37]` | `[1,1369,384]` | 37 x 37 image tokens |
| 7 | CLS concat | Host C++ | `[1,1369,384]` | `[1,1370,384]` | Adds one CLS token |
| 8 | Position add | Host C++ | `[1,1370,384]` | `[1,1370,384]` | FP32 elementwise Add |

The patch-projection dense-equivalent workload is approximately:

```text
37 x 37 x 588 x 384 = 0.309 GMAC
```

## 5. Encoder data flow

All 12 blocks preserve the logical token shape:

```text
Block input  [1,1370,384]
Block output [1,1370,384]
```

### 5.1 One Encoder Block

```text
x [1,1370,384]
 |
 | NPU SPU: LayerNorm core
 | gamma/beta are folded into QKV weights
 v
norm1 [1,1370,384]
 |
 | NPU: QKV Linear, A8 x B8 -> BF16
 v
qkv [1,1370,1152]
 |
 | Host C++ codec: split Q/K/V and 6 heads
 v
Q,K,V = 6 x [1,1370,64]
 |
 | NPU: fixed-scale INT8 QK + SPU softmax
 | NPU: dual-range probability + two INT8 AV MatMuls
 v
6 x context [1,1370,64]
 |
 | Host C++ codec: head concat
 v
attention context [1,1370,384]
 |
 | NPU: attention output projection
 v
attention projected [1,1370,384]
 |
 | Host C++ / resident path: Add with x
 v
post [1,1370,384]
 |
 | NPU SPU: LayerNorm core
 | gamma/beta are folded into FC1 weights
 v
norm2 [1,1370,384]
 |
 | NPU: FC1, 3 paired-output kernels
 v
fc1 [1,1370,1536]
 |
 | Host C++: exact FP32 GELU + quantize/pack
 v
gelu [1,1370,1536]
 |
 | NPU: FC2
 v
fc2 [1,1370,384]
 |
 | Host C++: residual Add with post
 v
y [1,1370,384]
```

### 5.2 Encoder operator allocation

| Module | Input | Output | Backend | Precision/allocation |
|---|---|---|---|---|
| Norm1 core | `[1,1370,384]` | Same | NPU SPU | BF16 normalization core |
| Norm1 affine | 384 gamma/beta values | Folded | NPU QKV weights | No separate affine dispatch |
| QKV projection | `[1,1370,384]` | `[1,1370,1152]` | NPU | A8 x B8 -> BF16 |
| Q/K/V and head split | `[1,1370,1152]` | 18 tensors of `[1,1370,64]` | Host C++ codec | Logical/layout operation |
| QK transpose MatMul | Per head `[1370,64] x [64,1370]` | Per head `[1370,1370]` logically | NPU | Executed by query slices |
| Softmax | Query slices of logits | Same | NPU SPU | Not performed on Host |
| Fine-probability AV | `[q,1370] x [1370,64]` | `[q,64]` | NPU | INT8 MatMul |
| Residual-probability AV | `[q,1370] x [1370,64]` | `[q,64]` | NPU | Second INT8 MatMul |
| Context merge | Two `[q,64]` outputs | `[q,64]` | NPU | BF16 unit-gain Add; no empirical gain |
| Attention projection | `[1,1370,384]` | Same | NPU | A8 x B8 -> BF16 |
| First residual Add | Two `[1,1370,384]` | `[1,1370,384]` | Host C++ / resident bridge | FP32 semantics |
| Norm2 core | `[1,1370,384]` | Same | NPU SPU | BF16 normalization core |
| Norm2 affine | 384 gamma/beta values | Folded | NPU FC1 weights | No separate affine dispatch |
| FC1 | `[1,1370,384]` | `[1,1370,1536]` | NPU | 3 kernels, each producing 512 channels |
| GELU | `[1,1370,1536]` | Same | Host C++ | Exact FP32 activation |
| FC2 | `[1,1370,1536]` | `[1,1370,384]` | NPU | A8 x B8 -> BF16 |
| Second residual Add | Two `[1,1370,384]` | `[1,1370,384]` | Host C++ | FP32 semantics |

### 5.3 Attention query slicing

For every head, K and V retain logical shape `[1370,64]`. The query rows are
processed as three two-output calls:

| Call | Q0 valid rows | Q1 valid rows | Q1 physical rows |
|---:|---|---|---:|
| 0 | `[0,256)` | `[256,512)` | 256 |
| 1 | `[512,768)` | `[768,1024)` | 256 |
| 2 | `[1024,1280)` | `[1280,1370)` | 256, of which 90 are valid |

One call conceptually executes:

```text
Q slice [256,64] x K^T [64,1370]
    -> logits [256,1370]
    -> SPU softmax
    -> fine A8 probability     [256,1370]
    -> residual A8 probability [256,1370]
    -> two AV MatMuls with V [1370,64]
    -> BF16 Add
    -> context [256,64]
```

The full `[1370,1370]` attention matrix is a logical tensor; the production
path does not need to materialize it as one contiguous Host tensor.

### 5.4 Block0 exception and decoder captures

Block0 is hybrid for accuracy:

| Block0 head | Allocation |
|---:|---|
| 0, 1, 2, 4, 5 | U250 NPU dual attention |
| 3 | Host FP32 QK, softmax, and AV |

Blocks 1 through 11 run all six attention heads on NPU with the fused
`attention6` entry point.

The Encoder always executes all 12 blocks. Outputs from Blocks 2, 5, 8, and 11
are retained as four Decoder inputs:

```text
Block0 -> Block1 -> Block2  ----> capture C1 [1,1370,384]
                    |
                    v
Block3 -> Block4 -> Block5  ----> capture C2 [1,1370,384]
                    |
                    v
Block6 -> Block7 -> Block8  ----> capture C3 [1,1370,384]
                    |
                    v
Block9 -> Block10 -> Block11 ----> capture C4 [1,1370,384]
```

These are feature-depth levels, not four different spatial resolutions. All
four have a 37x37 token grid; DPT creates the spatial pyramid during reassembly.

## 6. Decoder reassemble and scratch features

Each captured token tensor first follows the Host path:

```text
[1,1370,384]
  -> FP32 final LayerNorm
  -> Slice away CLS token
[1,1369,384]
  -> Transpose
[1,384,1369]
  -> Reshape
[1,384,37,37]
```

The four reassemble branches are:

```text
C1 from Block2 [1,384,37,37]
  -> NPU projects.0 1x1 Conv
     [1,48,37,37]
  -> NPU resize_layers.0 1x1 Conv
     [1,768,37,37]
  -> Host DepthToSpace x2
     [1,48,148,148]
  -> NPU layer1_rn 3x3 Conv
     L1 [1,64,148,148]

C2 from Block5 [1,384,37,37]
  -> NPU projects.1 1x1 Conv
     [1,96,37,37]
  -> NPU resize_layers.1 1x1 Conv
     [1,384,37,37]
  -> Host DepthToSpace
     [1,96,74,74]
  -> NPU layer2_rn 3x3 Conv
     L2 [1,64,74,74]

C3 from Block8 [1,384,37,37]
  -> NPU projects.2 1x1 Conv
     [1,192,37,37]
  -> NPU layer3_rn 3x3 Conv
     L3 [1,64,37,37]

C4 from Block11 [1,384,37,37]
  -> NPU projects.3 1x1 Conv
     [1,384,37,37]
  -> NPU resize_layers.3 3x3 stride-2 Conv
     [1,384,19,19]
  -> NPU layer4_rn 3x3 Conv
     L4 [1,64,19,19]
```

## 7. RefineNet shortcut semantics

### 7.1 Residual Convolution Unit

Each `resConfUnit` uses a same-shape identity shortcut:

```text
x [1,64,H,W]
 |----------------------------------------------+
 |                                              |
 +-> Host ReLU                                  |
     -> NPU Conv1 [64,64,3,3], stride 1, pad 1  |
     -> Host ReLU                               |
     -> NPU Conv2 [64,64,3,3], stride 1, pad 1  |
     -> Host Add <-------------------------------+
         y = x + Conv2(ReLU(Conv1(ReLU(x))))
```

No projection is needed on this shortcut because input and output are both
`[1,64,H,W]`.

### 7.2 Two shortcut classes

RefineNet3, RefineNet2, and RefineNet1 each contain:

1. An internal shortcut in `resConfUnit1`.
2. A cross-scale Add between the processed lateral feature and the top-down
   feature from the previous RefineNet.
3. An internal shortcut in `resConfUnit2`.

RefineNet4 has only `resConfUnit2`, because no deeper top-down input exists.

| RefineNet | resConfUnit1 shortcut | Cross-scale Add | resConfUnit2 shortcut | Total Add |
|---|---:|---:|---:|---:|
| RefineNet4 | 0 | 0 | 1 | 1 |
| RefineNet3 | 1 | 1 | 1 | 3 |
| RefineNet2 | 1 | 1 | 1 | 3 |
| RefineNet1 | 1 | 1 | 1 | 3 |
| Total | 3 | 3 | 4 | 10 |

## 8. RefineNet4 data flow

Inputs:

```text
L4 = layer4_rn output [1,64,19,19]
```

There is no `resConfUnit1` and no cross-scale fusion Add.

```text
L4 [1,64,19,19]
 |--------------------------------------------------+
 |                                                  |
 +-> Host ReLU                                      |
     -> NPU resConfUnit2.conv1, 3x3 64->64          |
        [1,64,19,19]                                |
     -> Host ReLU                                   |
     -> NPU resConfUnit2.conv2, 3x3 64->64          |
        [1,64,19,19]                                |
     -> Host Add <-----------------------------------+
        R4 [1,64,19,19]
     -> Host Resize 19x19 -> 37x37
        [1,64,37,37]
     -> NPU out_conv, 1x1 64->64
        T4 [1,64,37,37]
```

Formula:

```text
R4 = L4 + RCU2Conv(L4)
T4 = OutConv4(Resize(R4))
```

## 9. RefineNet3 data flow

Inputs:

```text
L3 = layer3_rn output          [1,64,37,37]
T4 = refinenet4 out_conv       [1,64,37,37]
```

```text
L3 [1,64,37,37]
 |--------------------------------------------------+
 |                                                  |
 +-> Host ReLU                                      |
     -> NPU resConfUnit1.conv1, 3x3 64->64          |
     -> Host ReLU                                   |
     -> NPU resConfUnit1.conv2, 3x3 64->64          |
     -> Host Add <-----------------------------------+
        R31 [1,64,37,37]

R31 [1,64,37,37] + T4 [1,64,37,37]
     -> Host cross-scale Add
        F3 [1,64,37,37]

F3 [1,64,37,37]
 |--------------------------------------------------+
 |                                                  |
 +-> Host ReLU                                      |
     -> NPU resConfUnit2.conv1, 3x3 64->64          |
     -> Host ReLU                                   |
     -> NPU resConfUnit2.conv2, 3x3 64->64          |
     -> Host Add <-----------------------------------+
        R32 [1,64,37,37]
     -> Host Resize 37x37 -> 74x74
        [1,64,74,74]
     -> NPU out_conv, 1x1 64->64
        T3 [1,64,74,74]
```

Formulas:

```text
R31 = L3 + RCU1Conv(L3)
F3  = R31 + T4
R32 = F3 + RCU2Conv(F3)
T3  = OutConv3(Resize(R32))
```

## 10. RefineNet2 data flow

Inputs:

```text
L2 = layer2_rn output          [1,64,74,74]
T3 = refinenet3 out_conv       [1,64,74,74]
```

```text
L2 [1,64,74,74]
 |--------------------------------------------------+
 |                                                  |
 +-> Host ReLU                                      |
     -> NPU resConfUnit1.conv1, 3x3 64->64          |
        [1,64,74,74]                                |
     -> Host ReLU                                   |
     -> NPU resConfUnit1.conv2, 3x3 64->64          |
        [1,64,74,74]                                |
     -> Host Add <-----------------------------------+
        R21 [1,64,74,74]

R21 [1,64,74,74] + T3 [1,64,74,74]
     -> Host cross-scale Add
        F2 [1,64,74,74]

F2 [1,64,74,74]
 |--------------------------------------------------+
 |                                                  |
 +-> Host ReLU                                      |
     -> NPU resConfUnit2.conv1, 3x3 64->64          |
        [1,64,74,74]                                |
     -> Host ReLU                                   |
     -> NPU resConfUnit2.conv2, 3x3 64->64          |
        [1,64,74,74]                                |
     -> Host Add <-----------------------------------+
        R22 [1,64,74,74]
     -> Host Resize 74x74 -> 148x148
        [1,64,148,148]
     -> NPU out_conv, 1x1 64->64
        T2 [1,64,148,148]
```

Formulas:

```text
R21 = L2 + RCU1Conv(L2)
F2  = R21 + T3
R22 = F2 + RCU2Conv(F2)
T2  = OutConv2(Resize(R22))
```

RefineNet2 contains four 3x3 NPU convolutions, one 1x1 NPU convolution, four
Host ReLUs, three Host Adds, and one Host Resize.

## 11. RefineNet1 data flow

Inputs:

```text
L1 = layer1_rn output          [1,64,148,148]
T2 = refinenet2 out_conv       [1,64,148,148]
```

```text
L1 [1,64,148,148]
 |--------------------------------------------------+
 |                                                  |
 +-> Host ReLU                                      |
     -> NPU resConfUnit1.conv1, 3x3 64->64          |
        [1,64,148,148]                              |
     -> Host ReLU                                   |
     -> NPU resConfUnit1.conv2, 3x3 64->64          |
        [1,64,148,148]                              |
     -> Host Add <-----------------------------------+
        R11 [1,64,148,148]

R11 [1,64,148,148] + T2 [1,64,148,148]
     -> Host cross-scale Add
        F1 [1,64,148,148]

F1 [1,64,148,148]
 |--------------------------------------------------+
 |                                                  |
 +-> Host ReLU                                      |
     -> NPU resConfUnit2.conv1, 3x3 64->64          |
        [1,64,148,148]                              |
     -> Host ReLU                                   |
     -> NPU resConfUnit2.conv2, 3x3 64->64          |
        [1,64,148,148]                              |
     -> Host Add <-----------------------------------+
        R12 [1,64,148,148]
     -> Host Resize 148x148 -> 296x296
        [1,64,296,296]
     -> NPU out_conv, 1x1 64->64
        T1 [1,64,296,296]
```

Formulas:

```text
R11 = L1 + RCU1Conv(L1)
F1  = R11 + T2
R12 = F1 + RCU2Conv(F1)
T1  = OutConv1(Resize(R12))
```

The four 148x148 3x3 convolutions are row tiled on U250. The package contains
first/middle/last tile variants; this shape executes two row tiles per source
Conv, selecting the applicable positional variants.

## 12. Depth output head

```text
T1 [1,64,296,296]
  -> NPU output_conv1, 3x3 64->32
     [1,32,296,296]
  -> Host Resize 296x296 -> 518x518
     [1,32,518,518]
  -> NPU output_conv2.0, 3x3 32->32
     [1,32,518,518]
  -> Host ReLU
     [1,32,518,518]
  -> NPU output_conv2.2, 1x1 32->1
     [1,1,518,518]
  -> Host ReLU + Squeeze
     depth [1,518,518]
```

The output is non-negative relative inverse depth. It is not metre-valued
depth until the App-side alignment described in Section 16.

## 13. Complete Decoder Conv inventory

All 32 source Conv modules below run on U250 NPU with calibrated A8 activations,
B8 weights, and BF16 outputs. `Variants/calls` reports compiled channel/tile
variants. Where a reusable set of positional tile variants is executed a
different number of times, the per-frame row-tile call count is shown in
parentheses. Neither value is the number of learned Conv layers.

| # | Conv module | Input shape | Weight shape | Output shape | Kernel / stride | Variants/calls |
|---:|---|---|---|---|---|---:|
| 1 | `projects.0` | `[1,384,37,37]` | `[48,384,1,1]` | `[1,48,37,37]` | 1x1 / 1 | 1 |
| 2 | `resize_layers.0.conv` | `[1,48,37,37]` | `[768,48,1,1]` | `[1,768,37,37]` | 1x1 / 1 | 12 |
| 3 | `projects.1` | `[1,384,37,37]` | `[96,384,1,1]` | `[1,96,37,37]` | 1x1 / 1 | 2 |
| 4 | `resize_layers.1.conv` | `[1,96,37,37]` | `[384,96,1,1]` | `[1,384,37,37]` | 1x1 / 1 | 6 |
| 5 | `projects.2` | `[1,384,37,37]` | `[192,384,1,1]` | `[1,192,37,37]` | 1x1 / 1 | 3 |
| 6 | `projects.3` | `[1,384,37,37]` | `[384,384,1,1]` | `[1,384,37,37]` | 1x1 / 1 | 6 |
| 7 | `resize_layers.3` | `[1,384,37,37]` | `[384,384,3,3]` | `[1,384,19,19]` | 3x3 / 2 | 6 |
| 8 | `layer1_rn` | `[1,48,148,148]` | `[64,48,3,3]` | `[1,64,148,148]` | 3x3 / 1 | 3 variants (2 row calls) |
| 9 | `layer2_rn` | `[1,96,74,74]` | `[64,96,3,3]` | `[1,64,74,74]` | 3x3 / 1 | 1 |
| 10 | `layer3_rn` | `[1,192,37,37]` | `[64,192,3,3]` | `[1,64,37,37]` | 3x3 / 1 | 1 |
| 11 | `layer4_rn` | `[1,384,19,19]` | `[64,384,3,3]` | `[1,64,19,19]` | 3x3 / 1 | 1 |
| 12 | `refinenet4.resConfUnit2.conv1` | `[1,64,19,19]` | `[64,64,3,3]` | `[1,64,19,19]` | 3x3 / 1 | 1 |
| 13 | `refinenet4.resConfUnit2.conv2` | `[1,64,19,19]` | `[64,64,3,3]` | `[1,64,19,19]` | 3x3 / 1 | 1 |
| 14 | `refinenet4.out_conv` | `[1,64,37,37]` | `[64,64,1,1]` | `[1,64,37,37]` | 1x1 / 1 | 1 |
| 15 | `refinenet3.resConfUnit1.conv1` | `[1,64,37,37]` | `[64,64,3,3]` | `[1,64,37,37]` | 3x3 / 1 | 1 |
| 16 | `refinenet3.resConfUnit1.conv2` | `[1,64,37,37]` | `[64,64,3,3]` | `[1,64,37,37]` | 3x3 / 1 | 1 |
| 17 | `refinenet3.resConfUnit2.conv1` | `[1,64,37,37]` | `[64,64,3,3]` | `[1,64,37,37]` | 3x3 / 1 | 1 |
| 18 | `refinenet3.resConfUnit2.conv2` | `[1,64,37,37]` | `[64,64,3,3]` | `[1,64,37,37]` | 3x3 / 1 | 1 |
| 19 | `refinenet3.out_conv` | `[1,64,74,74]` | `[64,64,1,1]` | `[1,64,74,74]` | 1x1 / 1 | 1 |
| 20 | `refinenet2.resConfUnit1.conv1` | `[1,64,74,74]` | `[64,64,3,3]` | `[1,64,74,74]` | 3x3 / 1 | 1 |
| 21 | `refinenet2.resConfUnit1.conv2` | `[1,64,74,74]` | `[64,64,3,3]` | `[1,64,74,74]` | 3x3 / 1 | 1 |
| 22 | `refinenet2.resConfUnit2.conv1` | `[1,64,74,74]` | `[64,64,3,3]` | `[1,64,74,74]` | 3x3 / 1 | 1 |
| 23 | `refinenet2.resConfUnit2.conv2` | `[1,64,74,74]` | `[64,64,3,3]` | `[1,64,74,74]` | 3x3 / 1 | 1 |
| 24 | `refinenet2.out_conv` | `[1,64,148,148]` | `[64,64,1,1]` | `[1,64,148,148]` | 1x1 / 1 | 1 |
| 25 | `refinenet1.resConfUnit1.conv1` | `[1,64,148,148]` | `[64,64,3,3]` | `[1,64,148,148]` | 3x3 / 1 | 3 variants (2 row calls) |
| 26 | `refinenet1.resConfUnit1.conv2` | `[1,64,148,148]` | `[64,64,3,3]` | `[1,64,148,148]` | 3x3 / 1 | 3 variants (2 row calls) |
| 27 | `refinenet1.resConfUnit2.conv1` | `[1,64,148,148]` | `[64,64,3,3]` | `[1,64,148,148]` | 3x3 / 1 | 3 variants (2 row calls) |
| 28 | `refinenet1.resConfUnit2.conv2` | `[1,64,148,148]` | `[64,64,3,3]` | `[1,64,148,148]` | 3x3 / 1 | 3 variants (2 row calls) |
| 29 | `refinenet1.out_conv` | `[1,64,296,296]` | `[64,64,1,1]` | `[1,64,296,296]` | 1x1 / 1 | 1 (8 row tiles) |
| 30 | `output_conv1` | `[1,64,296,296]` | `[32,64,3,3]` | `[1,32,296,296]` | 3x3 / 1 | 3 (4 row tiles) |
| 31 | `output_conv2.0` | `[1,32,518,518]` | `[32,32,3,3]` | `[1,32,518,518]` | 3x3 / 1 | 3 (7 row tiles) |
| 32 | `output_conv2.2` | `[1,32,518,518]` | `[1,32,1,1]` | `[1,1,518,518]` | 1x1 / 1 | 1 (7 row tiles) |

For a tiled 3x3 Conv, the input tile includes halo rows. The runtime removes
overlap from the tile outputs when reconstructing the logical output tensor.

## 14. Decoder Host operator inventory

The two backends alternate because all 32 Conv modules are on NPU while shape,
resize, activation, and residual operations are currently on Host.

| Host operator | Calls per inference | Main role |
|---|---:|---|
| LayerNormalization | 4 | Normalize the four captured Encoder tensors before reassembly |
| ReLU | 17 | Residual units and output head activation |
| Add | 10 | 7 internal shortcuts and 3 cross-scale fusions |
| Resize | 5 | Four RefineNet upscales plus final 296-to-518 resize |
| DepthToSpace | 3 | Reassemble Block2/Block5 features at 4x/2x resolution |
| Transpose | 4 | Token-to-feature conversion |
| Reshape | 4 | `[1369,384]` to `[384,37,37]` |
| Slice | 8 | Four CLS slices plus shape-control slices |
| Concat | 4 | Dynamic Resize shape construction |
| Shape | 4 | Dynamic Resize shape construction |
| Constant | 38 | Static graph-control values |
| Squeeze | 1 | `[1,1,518,518]` to `[1,518,518]` |

The important boundary pattern in every Residual Convolution Unit is:

```text
Host ReLU
  -> A8 quantize and native pack
  -> NPU Conv1
  -> BF16 unpack
  -> Host ReLU
  -> A8 quantize and native pack
  -> NPU Conv2
  -> BF16 unpack
  -> Host shortcut Add
```

This pattern explains why the Decoder remains sensitive to codec and dispatch
overhead even though all learned convolutions are allocated to NPU.

## 15. Precision and workload concentration

Current precision policy:

| Area | Policy |
|---|---|
| Patch projection | INT8 activation x B8 weight, BF16 output |
| Encoder Linear/MatMul | A8 x B8, BF16 output |
| Encoder LayerNorm core | NPU SPU BF16; affine folded into QKV/FC1 weights |
| Attention softmax | NPU SPU |
| Attention probability | Dual A8 fine/residual representation |
| Attention merge | BF16 Add, unit gain (`av_output_gain = 1.0`) |
| Encoder GELU | Host exact FP32 |
| Decoder Conv | A8 x B8, BF16 output |
| Decoder final LayerNorm, ReLU, Add, Resize, DepthToSpace | Host FP32 semantics |

Approximate dense-equivalent Encoder workload per block:

| NPU module | Approximate GMAC/block |
|---|---:|
| QKV | 0.606 |
| QK transpose | 0.721 |
| Fine AV | 0.721 |
| Residual AV | 0.721 |
| Attention output projection | 0.202 |
| FC1 | 0.808 |
| FC2 | 0.808 |
| Total | about 4.59 |

Twelve blocks account for about 55.1 dense-equivalent GMAC before Decoder
convolution workload. The majority of arithmetic is therefore on NPU. The main
Host workload is memory- and boundary-oriented:

- Twelve GELUs over `[1,1370,1536]`, about 25.25 million elements in total.
- Attention Q/K/V packing and physical-layout conversion.
- Block0 Head3 FP32 attention.
- Decoder ReLU/Add/Resize/DepthToSpace.
- Requantization, native pack/unpack, and dispatch at NPU/Host boundaries.

## 16. Model output and App metric-depth output

The direct neural-network output is:

```text
raw_relative_inverse [1,518,518]
```

The App resizes it to the Nebula411 depth grid and solves an affine mapping in
inverse-depth space over valid ToF pixels:

```text
inverse_metric = scale * raw_relative_inverse + shift
metric_depth   = 1 / clip(inverse_metric, 0.1, 10.0)
```

The resulting App tensor is normally `[600,800]` in metres. This postprocess is
outside the compiled Depth Anything model and runs on App CPU.

## 17. Implementation references

- App profile and live adapter:
  `portable_runtime/apps/completionformer_board_viewer/depthanything_u250.py`
- Hybrid U250 runtime:
  `.worktrees/u250-host-graph-optimization/tools/run_u250_depthanything_hybrid.py`
- Runtime-contract generator:
  `tools/generate_u250_runtime_contract.py`
- Model used to verify Decoder Conv attributes:
  `artifacts/depth_anything_v2_vits_tail_tokens_u250_a8b8_lnfold_boardcal_r39.onnx`

The board-side runtime contract is the authoritative source for the currently
deployed shapes, tiling, scales, and backend allocation. The ONNX model is used
to verify source Conv weight shapes, kernel sizes, padding, and strides.
