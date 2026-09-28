from pathlib import Path

import pytest
import torch

from ds_models import depth_anything_v2_vits_280 as adapter


def test_280_contract_matches_vit14_patch_grid():
    assert adapter.ifmap_sz == (3, 280, 280)
    assert 280 % 14 == 0
    assert (280 // 14) ** 2 + 1 == 401


def test_280_adapter_rejects_a_different_fixed_shape():
    if not Path(adapter.CHECKPOINT).is_file():
        pytest.skip("official Depth Anything checkpoint is external")
    model = adapter.Model()
    with pytest.raises(ValueError, match="requires input shape"):
        model(torch.zeros(1, 3, 518, 518))
