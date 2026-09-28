"""NumPy-only tensor codec for the DS ``npz2bin`` Python extension.

The vendor helper imports PyTorch only to convert float32 to/from BF16.  The
U250 runtime image does not ship PyTorch, so this module implements the same
round-to-nearest-even conversion directly on IEEE-754 bits.
"""

from pathlib import Path

import numpy as np


def readNPZFile(npz_data_path, target_key):
    path = Path(npz_data_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if target_key not in archive:
            raise KeyError(target_key)
        return {target_key: np.ascontiguousarray(archive[target_key])}


def _crop_to_dims(value, data_dims):
    array = np.asarray(value)
    dims = tuple(int(dim) for dim in data_dims)
    if array.ndim != len(dims):
        raise ValueError(f"tensor rank {array.ndim} does not match {len(dims)}")
    if any(actual < required for actual, required in zip(array.shape, dims)):
        raise ValueError(f"tensor shape {array.shape} is smaller than {dims}")
    if array.shape != dims:
        array = array[tuple(slice(0, dim) for dim in dims)]
    return np.ascontiguousarray(array)


def _float32_to_bf16_i16(value):
    array = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("BF16 input contains NaN or Inf")
    bits = array.view(np.uint32)
    # IEEE round-to-nearest-even: 0x7fff plus the retained low bit resolves
    # exact halfway cases toward an even BF16 significand.
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + bias) >> np.uint32(16)).astype(np.uint16)
    return np.ascontiguousarray(rounded.view(np.int16))


def _bf16_i16_to_float32(value):
    raw = np.ascontiguousarray(value, dtype=np.int16)
    bits = raw.view(np.uint16).astype(np.uint32) << np.uint32(16)
    result = np.ascontiguousarray(bits.view(np.float32))
    if not np.isfinite(result).all():
        raise ValueError("BF16 output contains NaN or Inf")
    return result


def createBF16TensorFromDict(data_dims, data_bw, npz_name, data):
    if npz_name not in data:
        raise KeyError(npz_name)
    result = _crop_to_dims(data[npz_name], data_dims)
    if int(data_bw) == 16:
        result = _float32_to_bf16_i16(result)
    elif int(data_bw) != 8:
        raise ValueError(f"unsupported tensor bitdepth {data_bw}")
    else:
        result = np.ascontiguousarray(result, dtype=np.int8)
    return result.astype(np.int32).reshape(-1).tolist()


def createBF16Tensor(data_dims, data_bw, npz_name, npz_data_path):
    return createBF16TensorFromDict(
        data_dims, data_bw, npz_name, readNPZFile(npz_data_path, npz_name)
    )


def exportNPZtoDict(dataArray, data_bitdepth, data_dims, data_name, exportint16=False):
    dims = tuple(int(dim) for dim in data_dims)
    if int(data_bitdepth) == 8:
        value = np.ascontiguousarray(dataArray, dtype=np.int8).reshape(dims)
        return {data_name: value}
    if int(data_bitdepth) != 16:
        raise ValueError(f"unsupported tensor bitdepth {data_bitdepth}")
    raw = np.ascontiguousarray(dataArray, dtype=np.int16).reshape(dims)
    decoded = _bf16_i16_to_float32(raw)
    return {
        data_name + "_i16": raw,
        data_name + "_bf16": decoded,
    }


def saveNPZFile(out_path, out_data):
    np.savez(out_path, **out_data)


def exportNPZ(dataArray, data_bitdepth, data_dims, data_name, out_path,
              exportint16=False):
    saveNPZFile(
        out_path,
        exportNPZtoDict(
            dataArray, data_bitdepth, data_dims, data_name, exportint16
        ),
    )
    return True
