from tools.compile_u250_base_kernels import FAMILIES
from tools.export_u250_base_kernels import SHAPES
from tools.package_u250_base import FAMILIES as PACKAGE_FAMILIES
from tools import npz_util

import numpy as np


def test_supported_shape_contracts_are_static_and_consistent():
    assert SHAPES == {
        280: {"tokens": 401, "patch_grid": 20},
        518: {"tokens": 1370, "patch_grid": 37},
    }
    for shape, contract in SHAPES.items():
        assert contract["patch_grid"] == shape // 14
        assert contract["tokens"] == contract["patch_grid"] ** 2 + 1


def test_compile_and_package_use_the_same_kernel_families():
    assert tuple(name for name, _ in FAMILIES) == PACKAGE_FAMILIES


def test_packaged_npz_helper_rounds_bf16_to_nearest_even():
    values = np.array([1.0, 1.00390625, 1.01171875], dtype=np.float32)
    encoded = np.asarray(
        npz_util.createBF16TensorFromDict([3], 16, "value", {"value": values}),
        dtype=np.int16,
    )
    decoded = npz_util.exportNPZtoDict(
        encoded, 16, [3], "value"
    )["value_bf16"]
    np.testing.assert_array_equal(decoded, np.array([1.0, 1.0, 1.015625]))
