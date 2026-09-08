"""Stable, address-independent U250 tensor layout descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class TensorLayoutDescriptor:
    layout: str
    dims: tuple[int, int, int, int]
    bitdepth: int
    c_align: int
    w_align: int
    combined_bytes: int
    direction: str
    index: int
    matrix_role: str

    @classmethod
    def from_tensor(cls, case_name: str, tensor: dict, direction: str,
                    index: int) -> "TensorLayoutDescriptor":
        try:
            layout = tensor["layout"]
            dims = tuple(int(value) for value in tensor["dims"])
            bitdepth = int(tensor["bitdepth"])
            c_align = int(tensor["c_align"])
            w_align = int(tensor["w_align"])
            combined_bytes = int(tensor["size_per_bank"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{case_name}: invalid tensor descriptor") from error

        if layout not in {"NCHW", "NDWC"}:
            raise ValueError(f"{case_name}: unsupported layout {layout!r}")
        if len(dims) != 4:
            raise ValueError(f"{case_name}: tensor dims must have rank four")
        if any(value <= 0 for value in dims):
            raise ValueError(f"{case_name}: tensor dims must be positive")
        if bitdepth not in {8, 16}:
            raise ValueError(f"{case_name}: unsupported bitdepth {bitdepth}")
        if c_align <= 0 or w_align <= 0:
            raise ValueError(f"{case_name}: tensor alignment must be positive")
        if direction not in {"input", "output"}:
            raise ValueError(f"{case_name}: invalid tensor direction {direction!r}")
        if combined_bytes % 256:
            raise ValueError(f"{case_name}: combined extent must be 256-byte aligned")

        if layout == "NCHW":
            matrix_role = "netio"
        elif direction == "output":
            matrix_role = "output"
        elif case_name.startswith("attention2") and index in {1, 2}:
            matrix_role = "right"
        else:
            matrix_role = "left"

        return cls(
            layout=layout,
            dims=dims,  # type: ignore[arg-type]
            bitdepth=bitdepth,
            c_align=c_align,
            w_align=w_align,
            combined_bytes=combined_bytes,
            direction=direction,
            index=index,
            matrix_role=matrix_role,
        )

    def identity(self) -> str:
        payload = asdict(self)
        payload.pop("index")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def storage_identity(self) -> str:
        """Identify valid physical lanes independently of cfg IO direction.

        The qualified native codec uses the same physical indexing for NDWC
        left/right/output tensors and for NCHW input/output tensors. Direction,
        tensor index, and matrix role constrain codec APIs, but do not alter the
        bytes consumed by a directly connected NPU kernel.
        """
        payload = asdict(self)
        for field in ("direction", "index", "matrix_role"):
            payload.pop(field)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def build_case_descriptors(
    records: dict[str, dict],
) -> dict[str, dict[str, list[TensorLayoutDescriptor]]]:
    return {
        case_name: {
            direction: [
                TensorLayoutDescriptor.from_tensor(case_name, tensor, direction, index)
                for index, tensor in enumerate(record[f"{direction}s"])
            ]
            for direction in ("input", "output")
        }
        for case_name, record in records.items()
    }
