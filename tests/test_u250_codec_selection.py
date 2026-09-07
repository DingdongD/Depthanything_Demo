from __future__ import annotations

from dataclasses import asdict, replace
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import u250_cpp_mapped_runtime as mapped
from tools import run_u250_depthanything_hybrid as runner
from tools.u250_layout_descriptors import TensorLayoutDescriptor


def descriptor(direction="input", index=0):
    return TensorLayoutDescriptor(
        "NDWC", (1, 1, 16, 16), 8, 1, 1, 256, direction, index,
        "left" if direction == "input" else "output",
    )


def descriptors():
    return {"case": {"input": [descriptor()], "output": [descriptor("output")]}}


def report_for(cases):
    unique = {d.identity(): d for directions in cases.values()
              for values in directions.values() for d in values}
    return {
        "manifest_sha256": "manifest-hash", "layout": "ALL",
        "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "descriptors": [
            {**asdict(d), "identity": identity, "native_exact": True,
             "pack_exact": True, "unpack_exact": True, "production_enabled": True,
             "benchmark": {"operation": "pack" if d.direction == "input" else "unpack",
                           "production_enabled": True, "native_median_ms": 1.0,
                           "vendor_median_ms": 2.0}}
            for identity, d in unique.items()
        ],
    }


def selection(tmp_path, *, mode="native", cases=None, report=None):
    cases = cases or descriptors()
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report if report is not None else report_for(cases)))
    return mapped.LayoutCodecSelection(mode, "manifest-hash", path, cases)


class CpuDma:
    """Only the external native codec/transport boundary is replaced."""
    events = []

    def __init__(self):
        self.events.append("transport")
        self.h2c = []
        self.programs = []

    @staticmethod
    def validate_descriptor(desc):
        CpuDma.events.append("validate:" + desc["direction"])

    @staticmethod
    def pack_tensor(value, desc):
        assert tuple(value.shape) == tuple(desc["dims"])
        flat = value.reshape(-1).view(np.uint8)
        return flat[:128].copy(), flat[128:].copy()

    @staticmethod
    def unpack_tensor(even, odd, desc):
        return np.concatenate((even, odd)).view(np.int8).reshape(desc["dims"])

    def h2c_batch_safe(self, requests):
        self.events.append("h2c")
        self.h2c.extend(requests)

    def run_npu_chain(self, programs, timeout_ms):
        self.events.append("npu")
        self.programs.extend(programs)
        return [0.001] * len(programs)

    def c2h_batch_safe(self, requests):
        self.events.append("c2h")
        return [np.full(size, i + 1, np.uint8) for i, (_, _, size) in enumerate(requests)]

    def stats(self):
        return {}

    def reset_stats(self):
        pass


def fake_extension(native_type=CpuDma, path=None):
    path = Path(path or __file__).resolve()
    return SimpleNamespace(DmaBatch=native_type, __file__=str(path),
                           _u250_extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture(autouse=True)
def clean_runtime_cache(monkeypatch):
    CpuDma.events = []
    mapped._RUNTIME_CACHE.clear()
    monkeypatch.setitem(sys.modules, "fpgaDmaBatch", fake_extension())
    yield
    mapped._RUNTIME_CACHE.clear()


def runtime(policy, native_type=CpuDma):
    return mapped.CppMappedRuntime(
        {"shared_fm_workspace_bytes": 16384}, fake_extension(native_type),
        codec_selection=policy,
    )


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.update(manifest_sha256="stale"), "manifest_sha256"),
    (lambda r: r.update(descriptors=[]), "not native_exact"),
    (lambda r: r["descriptors"][0].update(native_exact=False), "not native_exact"),
    (lambda r: r["descriptors"][0].update(native_exact="true"), "not native_exact"),
    (lambda r: r["descriptors"][0].update(direction="output"), "descriptor"),
    (lambda r: r["descriptors"][0].update(c_align=2), "descriptor"),
    (lambda r: r["descriptors"][0].update(dims=[True, 1, 16, 16]), "descriptor"),
    (lambda r: r["descriptors"][0].update(pack_exact=False), "pack_exact"),
    (lambda r: r["descriptors"][0].update(unpack_exact=False), "unpack_exact"),
    (lambda r: r["descriptors"][0].update(production_enabled=False), "production_enabled"),
    (lambda r: r["descriptors"][0].pop("production_enabled"), "production_enabled"),
    (lambda r: r["descriptors"][0]["benchmark"].update(production_enabled=False), "benchmark"),
    (lambda r: r["descriptors"][0]["benchmark"].update(operation="unpack"), "benchmark"),
    (lambda r: r["descriptors"][0]["benchmark"].update(native_median_ms=3.0), "benchmark"),
    (lambda r: r["descriptors"][0]["benchmark"].update(native_median_ms=float("nan")), "benchmark"),
    (lambda r: r["descriptors"][0]["benchmark"].update(vendor_median_ms=None), "benchmark"),
    (lambda r: r["descriptors"].append(dict(r["descriptors"][0])), "duplicate"),
])
def test_native_preflight_rejects_before_transport_or_bank(tmp_path, mutation, reason):
    report = report_for(descriptors())
    mutation(report)
    with pytest.raises(RuntimeError, match=reason) as error:
        device = runtime(selection(tmp_path, report=report))
        device.ensure_bank(np.zeros(256, np.uint8), "bank")
    assert "case" in str(error.value)
    assert CpuDma.events == []


def test_native_requires_report_even_before_extension_construction(tmp_path):
    with pytest.raises(RuntimeError, match="report"):
        runtime(mapped.LayoutCodecSelection("native", "manifest-hash", None, descriptors()))
    assert CpuDma.events == []


@pytest.mark.parametrize("content", ["not json", "[]", '{"manifest_sha256":"manifest-hash"}'])
def test_auto_malformed_report_falls_back_explicitly(tmp_path, content):
    path = tmp_path / "malformed.json"
    path.write_text(content)
    policy = mapped.LayoutCodecSelection("auto", "manifest-hash", path, descriptors())
    assert not policy.native_for("case", "input")
    assert "report" in policy.stats()["fallback_reasons"][descriptor().identity()]


def test_native_empty_active_set_does_not_bypass_required_report():
    with pytest.raises(RuntimeError, match="report"):
        runtime(mapped.LayoutCodecSelection("native", "manifest-hash", None, {}))
    assert CpuDma.events == []


def test_native_rejects_descriptor_registered_under_wrong_direction(tmp_path):
    cases = {"case": {"input": [descriptor("output")], "output": []}}
    with pytest.raises(RuntimeError, match="case.*direction"):
        runtime(selection(tmp_path, cases=cases))
    assert CpuDma.events == []


def test_native_preflight_checks_extension_before_opening_transport(tmp_path):
    class RejectingCodec(CpuDma):
        @staticmethod
        def validate_descriptor(desc):
            raise ValueError("unsupported stride")

    with pytest.raises(RuntimeError, match="case.*unsupported stride"):
        runtime(selection(tmp_path), RejectingCodec)
    assert CpuDma.events == []


def test_all_descriptor_validation_precedes_transport_and_bank(tmp_path):
    device = runtime(selection(tmp_path))
    device.ensure_bank(np.zeros(256, np.uint8), "bank")
    assert CpuDma.events == ["validate:input", "validate:output", "transport", "h2c"]


def test_auto_falls_back_entire_case_direction_and_explains_companions(tmp_path):
    cases = descriptors()
    cases["case"]["input"].append(replace(descriptor(index=1), matrix_role="right"))
    report = report_for(cases)
    report["descriptors"][1]["native_exact"] = False
    policy = selection(tmp_path, mode="auto", cases=cases, report=report)
    assert policy.native_for("case", "input") is False
    assert policy.native_for("case", "output") is True
    reasons = policy.stats()["fallback_reasons"]
    assert reasons[cases["case"]["input"][1].identity()] == "not native_exact"
    assert "case input batch" in reasons[descriptor().identity()]


def test_auto_missing_report_falls_back_with_reason():
    policy = mapped.LayoutCodecSelection("auto", "manifest-hash", None, descriptors())
    assert policy.native_for("case", "input") is False
    assert "report" in policy.stats()["fallback_reasons"][descriptor().identity()]


def test_vendor_ignores_report_and_needs_no_native_extension_api(tmp_path):
    policy = mapped.LayoutCodecSelection("vendor", "manifest-hash", tmp_path / "absent", descriptors())
    device = runtime(policy, type("OldDma", (CpuDma,), {"validate_descriptor": None,
                                                     "pack_tensor": None, "unpack_tensor": None}))
    assert device.stats()["layout_codec"] == "vendor"
    assert CpuDma.events == ["transport"]
    assert policy.stats()["fallback_reasons"] == {}


def test_native_methods_account_logical_and_physical_bytes_and_reset(tmp_path):
    policy = selection(tmp_path)
    device = runtime(policy)
    value = np.arange(256, dtype=np.uint8).view(np.int8).reshape(1, 1, 16, 16)
    banks = device.pack_tensor(value, descriptor())
    decoded = device.unpack_tensor(*banks, descriptor("output"))
    np.testing.assert_array_equal(decoded, value)
    stats = device.stats()
    assert stats["native_pack_calls"] == stats["native_unpack_calls"] == 1
    assert stats["vendor_pack_calls"] == stats["vendor_unpack_calls"] == 0
    assert stats["native_pack_logical_bytes"] == stats["native_pack_physical_bytes"] == 256
    assert stats["native_unpack_logical_bytes"] == stats["native_unpack_physical_bytes"] == 256
    assert stats["codec_pack_ms_total"] == stats["native_pack_ms"] >= 0
    assert stats["codec_unpack_ms_total"] == stats["native_unpack_ms"] >= 0
    assert stats["codec_by_layout_dtype"]["NDWC_INT8"]["native_pack_calls"] == 1
    device.reset_frame_stats()
    assert device.stats()["native_pack_calls"] == 0


def test_native_stats_reject_any_vendor_accounting(tmp_path):
    policy = selection(tmp_path)
    with pytest.raises(RuntimeError, match="vendor"):
        policy.record("vendor", "pack", [descriptor()], [256], 1.0)
        policy.stats()


def test_bf16_accounting_distinguishes_float32_logical_bytes_from_physical_bytes(tmp_path):
    desc = replace(descriptor("output"), bitdepth=16, c_align=2, w_align=2, combined_bytes=512)
    policy = selection(tmp_path, mode="vendor", cases={"case": {"input": [], "output": [desc]}})
    policy.record("vendor", "unpack", [desc], [1024], 5.0)
    stats = policy.stats()
    assert stats["vendor_unpack_logical_bytes"] == 1024
    assert stats["vendor_unpack_physical_bytes"] == 512
    assert stats["codec_by_layout_dtype"]["NDWC_BF16"]["vendor_unpack_ms"] == 5.0


def test_cached_transport_revalidates_policy_before_bank_access(tmp_path, monkeypatch):
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: fake_extension())
    manifest = {"shared_fm_workspace_bytes": 16384}
    first, reused = mapped.get_cached_cpp_runtime("case", manifest, None, safe_dma=True,
                                                codec_selection=selection(tmp_path))
    assert reused is False
    first.ensure_bank(np.zeros(256, np.uint8), "bank")
    next_policy = selection(tmp_path, mode="vendor")
    second, reused = mapped.get_cached_cpp_runtime("case", manifest, None, safe_dma=True,
                                                 codec_selection=next_policy)
    assert second is first and reused is True
    assert second.stats()["layout_codec"] == "vendor"
    assert CpuDma.events.count("transport") == 1


def test_cached_vendor_transport_cannot_bypass_native_preflight(tmp_path, monkeypatch):
    class OldDma(CpuDma):
        validate_descriptor = None

    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: fake_extension(OldDma))
    manifest = {"shared_fm_workspace_bytes": 16384}
    first, _ = mapped.get_cached_cpp_runtime("case", manifest, None, safe_dma=True,
                                             codec_selection=selection(tmp_path, mode="vendor"))
    before = list(CpuDma.events)
    with pytest.raises(RuntimeError, match="case.*native codec API"):
        second, _ = mapped.get_cached_cpp_runtime("case", manifest, None, safe_dma=True,
                                                  codec_selection=selection(tmp_path))
        second.ensure_bank(np.zeros(256, np.uint8), "bank")
    assert first.codec_selection.mode == "vendor"
    assert CpuDma.events == before


class VendorCodec:
    def __init__(self):
        self.pack_calls = self.unpack_calls = 0

    def read_npz_dict(self, callback, name, values):
        self.pack_calls += 1
        return [next(iter(value.values())).reshape(-1).view(np.uint8) for value in values]

    def buffer_to_npz_dict(self, callback, name, values):
        self.unpack_calls += 1
        return [{"output": value.view(np.int8).reshape(1, 1, 16, 16)} for value in values]


@pytest.mark.parametrize("digest,reason", [(None, "extension_sha256"),
    ("bad", "extension_sha256"), (123, "extension_sha256"),
    ("0" * 64, "extension.*mismatch")])
def test_native_extension_digest_rejected_before_module_or_transport(tmp_path, monkeypatch,
                                                                  digest, reason):
    report = report_for(descriptors())
    if digest is None:
        report.pop("extension_sha256")
    else:
        report["extension_sha256"] = digest
    def load(path):
        CpuDma.events.append("module")
        return fake_extension()
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", load)
    with pytest.raises(RuntimeError, match=reason):
        device, _ = mapped.get_cached_cpp_runtime("case", {"shared_fm_workspace_bytes": 16384},
            Path(__file__), safe_dma=True, codec_selection=selection(tmp_path, report=report))
        device.ensure_bank(np.zeros(256, np.uint8), "bank")
    assert CpuDma.events == []


def test_correct_extension_digest_is_recorded_before_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: fake_extension())
    device, _ = mapped.get_cached_cpp_runtime("case", {"shared_fm_workspace_bytes": 16384},
        Path(__file__), safe_dma=True, codec_selection=selection(tmp_path))
    assert device.stats()["extension_sha256"] == report_for(descriptors())["extension_sha256"]
    assert device.stats()["extension_path"] == str(Path(__file__).resolve())
    assert CpuDma.events == ["validate:input", "validate:output", "transport"]


@pytest.mark.parametrize("initial_mode", ["vendor", "native"])
@pytest.mark.parametrize("change", ["file_replacement", "different_path", "different_bytes"])
def test_cached_extension_change_rejects_before_reuse(tmp_path, monkeypatch, initial_mode, change):
    extension_path = tmp_path / "fpgaDmaBatch.so"
    extension_path.write_bytes(b"qualified extension")
    extension = fake_extension(path=extension_path)
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: extension)
    report = report_for(descriptors())
    report["extension_sha256"] = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    manifest = {"shared_fm_workspace_bytes": 16384}
    device, _ = mapped.get_cached_cpp_runtime("case", manifest, extension_path, safe_dma=True,
        codec_selection=selection(tmp_path, mode=initial_mode, report=report))
    before = list(CpuDma.events)
    if change == "file_replacement":
        extension_path.write_bytes(b"replacement extension")
    else:
        extension_path = tmp_path / "other.so"
        extension_path.write_bytes(b"qualified extension" if change == "different_path"
                                   else b"different extension")
    # Even a new report matching the requested bytes cannot qualify a stale module.
    report["extension_sha256"] = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="extension provenance.*(changed|mismatch)"):
        next_device, _ = mapped.get_cached_cpp_runtime("case", manifest, extension_path,
            safe_dma=True, codec_selection=selection(tmp_path, report=report))
        next_device.ensure_bank(np.zeros(256, np.uint8), "bank")
    assert device.codec_selection.mode == initial_mode
    assert CpuDma.events == before


def test_cached_vendor_to_native_with_matching_digest_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: fake_extension())
    manifest = {"shared_fm_workspace_bytes": 16384}
    first, _ = mapped.get_cached_cpp_runtime("case", manifest, Path(__file__), safe_dma=True,
        codec_selection=selection(tmp_path, mode="vendor"))
    second, reused = mapped.get_cached_cpp_runtime("case", manifest, Path(__file__), safe_dma=True,
        codec_selection=selection(tmp_path))
    assert second is first and reused
    assert second.codec_selection.native_for("case", "input")
    assert CpuDma.events == ["transport", "validate:input", "validate:output"]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("digest", [None, "bad", "0" * 64])
def test_auto_extension_mismatch_falls_back_entire_directions_and_accounts(tmp_path, monkeypatch, cached, digest):
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: fake_extension())
    manifest = {"shared_fm_workspace_bytes": 16384}
    if cached:
        mapped.get_cached_cpp_runtime("case", manifest, Path(__file__), safe_dma=True,
            codec_selection=selection(tmp_path, mode="vendor"))
    cases = descriptors()
    cases["case"]["input"].append(replace(descriptor(index=1), matrix_role="right"))
    report = report_for(cases)
    if digest is None:
        report.pop("extension_sha256")
    else:
        report["extension_sha256"] = digest
    policy = selection(tmp_path, mode="auto", cases=cases, report=report)
    device, reused = mapped.get_cached_cpp_runtime("case", manifest, Path(__file__), safe_dma=True,
                                                 codec_selection=policy)
    assert reused is cached
    assert not policy.native_for("case", "input")
    assert not policy.native_for("case", "output")
    reasons = device.stats()["fallback_reasons"]
    assert len(reasons) == 3 and len(set(reasons.values())) == 1
    assert "extension provenance" in next(iter(reasons.values()))
    codec, vendor = codecs(policy, device)
    values = [np.zeros((1, 1, 16, 16), np.int8)] * 2
    codec.pack_inputs("case", values)
    codec.decode_outputs("case", [np.zeros(256, np.uint8)])
    stats = device.stats()
    assert stats["native_pack_calls"] == stats["native_unpack_calls"] == 0
    assert stats["vendor_pack_calls"] == 2 and stats["vendor_unpack_calls"] == 1
    assert CpuDma.events == ["transport"]


def test_untracked_loaded_module_cannot_gain_native_eligibility(tmp_path):
    extension = fake_extension()
    del extension._u250_extension_sha256
    with pytest.raises(RuntimeError, match="extension provenance"):
        mapped.CppMappedRuntime({"shared_fm_workspace_bytes": 16384}, extension,
                               codec_selection=selection(tmp_path))
    assert CpuDma.events == []


@pytest.mark.parametrize("change", ["replacement", "requested_path"])
def test_auto_cached_extension_change_falls_back_before_native_api(tmp_path, monkeypatch, change):
    path = tmp_path / "fpgaDmaBatch.so"
    path.write_bytes(b"original extension")
    extension = fake_extension(path=path)
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: extension)
    manifest = {"shared_fm_workspace_bytes": 16384}
    first, _ = mapped.get_cached_cpp_runtime("case", manifest, path, safe_dma=True,
        codec_selection=selection(tmp_path, mode="vendor"))
    if change == "requested_path":
        path = tmp_path / "other.so"
    path.write_bytes(b"different extension")
    report = report_for(descriptors())
    report["extension_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    device, reused = mapped.get_cached_cpp_runtime("case", manifest, path, safe_dma=True,
        codec_selection=selection(tmp_path, mode="auto", report=report))
    assert device is first and reused
    assert not device.codec_selection.native_for("case", "input")
    assert not device.codec_selection.native_for("case", "output")
    assert all("extension provenance" in reason for reason in device.stats()["fallback_reasons"].values())
    assert device.extension_sha256 != report["extension_sha256"]
    assert CpuDma.events == ["transport"]


def test_loader_records_digest_before_import_and_rejects_replacement_without_runtime_cache(tmp_path, monkeypatch):
    path = tmp_path / "fpgaDmaBatch.py"
    path.write_text("from tools.u250_cpp_mapped_runtime import _extension_sha256\n"
                    "from pathlib import Path\n"
                    "loaded_bytes = _extension_sha256(Path(__file__))\n")
    extension = mapped.load_fpga_dma_batch(path)
    assert extension._u250_extension_sha256 == extension.loaded_bytes
    monkeypatch.setattr(extension, "DmaBatch", CpuDma, raising=False)
    before = list(CpuDma.events)
    path.write_text("raise AssertionError('must not import replacement')\n")
    report = report_for(descriptors())
    report["extension_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="extension provenance.*changed"):
        mapped.get_cached_cpp_runtime("other-case", {"shared_fm_workspace_bytes": 16384}, path,
            safe_dma=True, codec_selection=selection(tmp_path, report=report))
    assert CpuDma.events == before


def test_loader_returning_another_compatible_module_rejects_before_transport(tmp_path, monkeypatch):
    path = tmp_path / "other.so"
    path.write_bytes(b"another API-compatible extension")
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: fake_extension(path=path))
    with pytest.raises(RuntimeError, match="extension provenance.*path changed"):
        mapped.get_cached_cpp_runtime("case", {"shared_fm_workspace_bytes": 16384}, Path(__file__),
            safe_dma=True, codec_selection=selection(tmp_path))
    assert CpuDma.events == []


def codecs(policy, device):
    vendor = VendorCodec()
    registry = SimpleNamespace(descriptors=policy.descriptors, npz2bin=vendor, quiet=False,
                               activate=lambda name: None)
    return runner.RuntimeTensorCodec(registry, policy, device, None, None,
                                     lambda value: "output"), vendor


def test_runner_native_pack_and_decode_bypass_vendor(tmp_path):
    policy = selection(tmp_path)
    codec, vendor = codecs(policy, runtime(policy))
    logical = np.arange(256, dtype=np.uint8).view(np.int8).reshape(1, 1, 16, 16)
    physical = codec.pack_inputs("case", [logical])
    restored = codec.decode_outputs("case", physical)
    np.testing.assert_array_equal(restored[0], logical)
    assert isinstance(physical[0], tuple)
    assert vendor.pack_calls == vendor.unpack_calls == 0


@pytest.mark.parametrize("operation", ["pack", "unpack"])
def test_native_runtime_error_includes_tensor_context(tmp_path, monkeypatch, operation):
    policy = selection(tmp_path)
    device = runtime(policy)
    codec, vendor = codecs(policy, device)

    def invalid_tensor(*args):
        raise ValueError("invalid dtype or bank extent")

    monkeypatch.setattr(device, f"{operation}_tensor", invalid_tensor)
    direction = "input" if operation == "pack" else "output"
    identity = descriptor(direction).identity()
    with pytest.raises(RuntimeError, match=f"case: {direction} 0 descriptor {identity}.*invalid dtype"):
        if operation == "pack":
            codec.pack_inputs("case", [np.zeros((1, 1, 16, 16), np.int8)])
        else:
            codec.decode_outputs("case", [(np.zeros(128, np.uint8), np.zeros(128, np.uint8))])
    assert vendor.pack_calls == vendor.unpack_calls == 0


def test_runner_auto_vendor_fallback_accounts_every_tensor(tmp_path):
    cases = descriptors()
    cases["case"]["input"].append(replace(descriptor(index=1), matrix_role="right"))
    report = report_for(cases)
    report["descriptors"][1]["production_enabled"] = False
    policy = selection(tmp_path, mode="auto", cases=cases, report=report)
    codec, vendor = codecs(policy, runtime(policy))
    logical = np.zeros((1, 1, 16, 16), np.int8)
    physical = codec.pack_inputs("case", [logical, logical])
    assert all(isinstance(value, np.ndarray) for value in physical)
    assert vendor.pack_calls == 1
    stats = policy.stats()
    assert stats["vendor_pack_calls"] == 2
    assert stats["vendor_pack_logical_bytes"] == stats["vendor_pack_physical_bytes"] == 512
    assert stats["native_pack_calls"] == 0


def test_vendor_codec_preserves_logical_output_and_aggregate_totals(tmp_path):
    policy = selection(tmp_path, mode="vendor")
    codec, vendor = codecs(policy, runtime(policy))
    value = np.arange(256, dtype=np.uint8).view(np.int8).reshape(1, 1, 16, 16)
    restored = codec.decode_outputs("case", codec.pack_inputs("case", [value]))
    np.testing.assert_array_equal(restored[0], value)
    assert vendor.pack_calls == vendor.unpack_calls == 1
    stats = policy.stats()
    assert stats["vendor_pack_calls"] == stats["vendor_unpack_calls"] == 1
    assert stats["codec_pack_ms_total"] == stats["vendor_pack_ms"]
    assert stats["codec_unpack_ms_total"] == stats["vendor_unpack_ms"]


def test_active_case_collection_respects_capture_and_resume_modes():
    block = {"host_norm1": {"npu_core": "norm1"}, "host_norm2": {},
             "qkv": {"kernel": "qkv"}, "attention": {"heads": [{"kernel": "attn"}]},
             "post_attention": {"kernel": "post"},
             "mlp": {"fc1_kernels": ["fc1"], "fc2_kernel": "fc2"}}
    contract = {"frontend": {}, "encoder": [block], "decoder": [
        {"backend": "npu_layernorm", "source_node": "ln", "kernel": "decoder_norm"}]}
    plan = {"frontend": {"projection_kernels": ["patch"]}, "decoder_steps": [
        {"name": "ln", "backend": "host"},
        {"name": "conv", "backend": "npu", "kernels": [{"name": "conv0"}]}]}
    args = SimpleNamespace(encoder_captures=None, encoder_resume=None, encoder_start_layer=None)
    assert runner.active_codec_cases(contract, plan, args) == {
        "patch", "norm1", "qkv", "attn", "post", "fc1", "fc2", "decoder_norm", "conv0"}
    args.encoder_captures = "captures.npz"
    assert runner.active_codec_cases(contract, plan, args) == {"decoder_norm", "conv0"}
    args.encoder_captures = None
    args.encoder_resume = "resume.npz"
    args.encoder_start_layer = 1
    assert runner.active_codec_cases(contract, plan, args) == {"decoder_norm", "conv0"}


def grouped_record():
    return {"name": "case", "base_addresses": [10, 20, 30, 40, 1000, 50],
            "isa_ranges": [2, 1], "inputs": [{"address": 0, "size_per_bank": 256}],
            "outputs": [{"address": 4, "size_per_bank": 256}]}


@pytest.mark.parametrize("upload", [True, False])
def test_native_group_passes_bank_pairs_without_split_or_merge(tmp_path, monkeypatch, upload):
    device = runtime(selection(tmp_path))
    banks = (np.full(128, 5, np.uint8), np.full(128, 9, np.uint8))

    def forbidden(*args):
        pytest.fail("native tensor invoked Python combined-buffer conversion")

    monkeypatch.setattr(mapped, "split_combined_ddr", forbidden)
    monkeypatch.setattr(mapped, "merge_combined_ddr", forbidden)
    outputs, timing = device.run_group([grouped_record()], [[banks]], [[upload]], 1234)
    assert isinstance(outputs[0][0], tuple)
    np.testing.assert_array_equal(outputs[0][0][0], np.full(128, 1, np.uint8))
    np.testing.assert_array_equal(outputs[0][0][1], np.full(128, 2, np.uint8))
    assert timing["h2c_bytes"] == (256 if upload else 0)
    assert timing["h2c_skipped_bytes"] == (0 if upload else 256)
    assert timing["c2h_bytes"] == 256
    if upload:
        assert device.transport.h2c[0][2] is banks[0]
        assert device.transport.h2c[1][2] is banks[1]
    else:
        assert device.transport.h2c == []


@pytest.mark.parametrize("packed", [np.zeros(256, np.uint8),
                                     (np.zeros(127, np.uint8), np.zeros(129, np.uint8)),
                                     (np.zeros(128, np.int8), np.zeros(128, np.uint8))])
def test_native_group_rejects_non_native_input_before_transfer(tmp_path, packed):
    device = runtime(selection(tmp_path))
    before = list(CpuDma.events)
    with pytest.raises(ValueError, match="bank"):
        device.run_group([grouped_record()], [[packed]], [[True]], 1234)
    assert CpuDma.events == before


def test_auto_keeps_native_input_pairs_when_output_falls_back(tmp_path, monkeypatch):
    report = report_for(descriptors())
    report["descriptors"][1]["production_enabled"] = False
    device = runtime(selection(tmp_path, mode="auto", report=report))
    monkeypatch.setattr(mapped, "split_combined_ddr", lambda *_: pytest.fail("split native banks"))
    outputs, _ = device.run_group([grouped_record()],
                                 [[(np.zeros(128, np.uint8), np.zeros(128, np.uint8))]],
                                 [[True]], 1234)
    assert isinstance(outputs[0][0], np.ndarray)
    assert outputs[0][0].shape == (256,)


@pytest.fixture
def runner_package(tmp_path, monkeypatch):
    """A one-kernel package exercises real CLI, cfg, policy, and DMA assembly."""
    runner._CFG_REGISTRY_CACHE.clear()
    monkeypatch.setattr(runner, "_NPZ_YAML_PATHS", None)
    cfg = ("Address: 0 (0x0) Size: 256 Layout: NDWC Dims: [1, 1, 16, 16] "
           "c_align: 1 w_align: 1 bitdepth: 8\n"
           "Output Address: 4 (0x4) Size: 256 Layout: NDWC Dims: [1, 1, 16, 16] "
           "c_align: 1 w_align: 1 bitdepth: 8\n")
    (tmp_path / "case_cfg.txt").write_text(cfg)
    record = grouped_record()
    for direction in ("input", "output"):
        record[direction + "s"][0].update(layout="NDWC", dims=[1, 1, 16, 16], bitdepth=8)
    manifest = {"cases": [record], "shared_fm_workspace_bytes": 16384,
                "bank_file": "bank.bin", "bank_sha256": "bank-hash"}
    contract = {"encoder": [], "decoder": [{"source_node": "conv", "backend": "npu",
                "input_quantization": {"scale": 1.0}, "kernels": [{"name": "case"}]}]}
    plan = {"encoder": [], "capture_layers": [], "capture_tensor_names": [],
            "model_outputs": ["out"], "decoder_steps": [
                {"name": "conv", "backend": "npu", "input_scale": 1.0,
                 "inputs": ["source"], "outputs": ["out"], "index": 0,
                 "row_tiles": 1, "kernels": [{"name": "case"}]}]}
    for name, value in (("manifest", manifest), ("contract", contract), ("plan", plan)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    report = report_for(descriptors())
    report["manifest_sha256"] = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "bank.bin").write_bytes(bytes(256))
    np.savez(tmp_path / "params.npz", source=np.zeros((1, 1, 16, 16), np.float32))
    np.save(tmp_path / "input.npy", np.zeros((1, 1, 16, 16), np.float32))
    vendor = VendorCodec()
    vendor.read_yaml = lambda _: None
    vendor.read_cfg = lambda _: None
    monkeypatch.setitem(sys.modules, "npz2bin", vendor)
    monkeypatch.setitem(sys.modules, "fpgaDma", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "npz_util", SimpleNamespace(createBF16TensorFromDict=None))
    support = {name: None for name in (
        "C2H_DEVICES", "H2C_DEVICES", "EventWaiter", "clear_interrupt", "configure_npu",
        "export_npz_allow_nonfinite", "reg_write", "tensor_metrics")}
    support.update(DDR_BASES=mapped.DDR_BASES, merge_2ddr=mapped.merge_combined_ddr,
                   split_2ddr=mapped.split_combined_ddr,
                   preferred_output_key=lambda _: "output",
                   sha256_array=lambda value: hashlib.sha256(value.tobytes()).hexdigest())
    monkeypatch.setitem(sys.modules, "run_u250_resident_compiled_case", SimpleNamespace(**support))
    monkeypatch.setattr(mapped, "load_fpga_dma_batch", lambda _: fake_extension())
    original_ensure = mapped.CppMappedRuntime.ensure_bank

    def ensure_bank(self, *args):
        CpuDma.events.append("ensure_bank")
        return original_ensure(self, *args)

    monkeypatch.setattr(mapped.CppMappedRuntime, "ensure_bank", ensure_bank)
    args = ["runner", "--case-dir", str(tmp_path), "--runtime-dir", str(tmp_path),
            "--manifest", str(tmp_path / "manifest.json"), "--contract", str(tmp_path / "contract.json"),
            "--host-plan", str(tmp_path / "plan.json"), "--host-params", str(tmp_path / "params.npz"),
            "--cfg-dir", str(tmp_path), "--input", str(tmp_path / "input.npy"),
            "--output", str(tmp_path / "output.npz"), "--depth-only"]
    yield args, report, vendor
    runner._CFG_REGISTRY_CACHE.clear()


def test_runner_native_preflight_runs_before_ensure_bank(tmp_path, monkeypatch, runner_package):
    args, report, _ = runner_package
    report["descriptors"][1]["unpack_exact"] = False
    (tmp_path / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", args + ["--layout-codec", "native", "--layout-codec-report",
                                          str(tmp_path / "report.json")])
    with pytest.raises(RuntimeError, match="unpack_exact"):
        runner.main()
    assert CpuDma.events == []


@pytest.mark.parametrize("change", ["manifest_geometry", "manifest_address", "cfg_geometry",
                                    "cfg_alignment", "manifest_and_cfg_geometry"])
def test_cached_cfg_changes_reject_native_before_transport(tmp_path, monkeypatch, runner_package, change):
    args, report, vendor = runner_package
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original, reused = runner.get_cached_cfg_registry(
        tmp_path, {item["name"]: item for item in manifest["cases"]}, vendor, quiet=False)
    assert reused is False
    cfg_path = tmp_path / "case_cfg.txt"
    if change in {"manifest_geometry", "manifest_and_cfg_geometry"}:
        manifest["cases"][0]["inputs"][0]["dims"] = [1, 1, 8, 32]
    if change == "manifest_address":
        manifest["cases"][0]["inputs"][0]["address"] = 1
    if change in {"cfg_geometry", "manifest_and_cfg_geometry"}:
        cfg_path.write_text(cfg_path.read_text().replace("[1, 1, 16, 16]", "[1, 1, 8, 32]", 1))
    if change == "cfg_alignment":
        cfg_path.write_text(cfg_path.read_text().replace("c_align: 1", "c_align: 2", 1))
    manifest_path.write_text(json.dumps(manifest))
    report["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", args + ["--layout-codec", "native", "--layout-codec-report",
                                          str(tmp_path / "report.json")])
    with pytest.raises(RuntimeError, match="cfg registry.*changed"):
        runner.main()
    assert CpuDma.events == []
    assert runner._CFG_REGISTRY_CACHE[str(tmp_path.resolve())] is original


def test_cfg_cache_reuses_unchanged_inputs_independent_of_mapping_order(tmp_path, runner_package):
    _, _, vendor = runner_package
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    records = {item["name"]: item for item in manifest["cases"]}
    first, reused = runner.get_cached_cfg_registry(tmp_path, records, vendor, quiet=False)
    assert reused is False
    for record in records.values():
        for tensor in record["inputs"] + record["outputs"]:
            reordered = dict(reversed(list(tensor.items())))
            tensor.clear()
            tensor.update(reordered)
    second, reused = runner.get_cached_cfg_registry(tmp_path, records, vendor, quiet=False)
    assert second is first and reused is True


@pytest.mark.parametrize("mode", ["vendor", "native"])
def test_pack_timing_includes_contiguous_input_preparation(tmp_path, monkeypatch, mode):
    policy = selection(tmp_path, mode=mode)
    device = runtime(policy)
    codec, _ = codecs(policy, device)
    logical = np.arange(256, dtype=np.uint8).view(np.int8).reshape(1, 1, 16, 16).swapaxes(2, 3)
    assert not logical.flags.c_contiguous
    clock = [0.0]
    original_contiguous = np.ascontiguousarray

    def contiguous(value, *args, **kwargs):
        if value is logical:
            clock[0] += 0.007  # A known seven-millisecond input preparation cost.
        return original_contiguous(value, *args, **kwargs)

    monkeypatch.setattr(runner.np, "ascontiguousarray", contiguous)
    monkeypatch.setattr(mapped.time, "perf_counter", lambda: clock[0])
    codec.pack_inputs("case", [logical])
    stats = policy.stats()
    assert clock[0] == pytest.approx(0.007)
    assert stats[f"{mode}_pack_calls"] == 1
    assert stats[f"{mode}_pack_ms"] == pytest.approx(7.0)
    assert stats["codec_pack_ms_total"] == pytest.approx(7.0)


@pytest.mark.parametrize("mode", ["vendor", "auto", "native"])
def test_runner_cli_routes_and_publishes_compatible_totals(tmp_path, monkeypatch, runner_package, mode):
    args, report, vendor = runner_package
    monkeypatch.setattr(sys, "argv", args + ["--layout-codec", mode, "--layout-codec-report",
                                          str(tmp_path / "report.json")])
    assert runner.main() == 0
    summary = json.loads((tmp_path / "output.summary.json").read_text())
    expected = np.concatenate((np.ones(128, np.int8), np.full(128, 2, np.int8))).reshape(1, 1, 16, 16)
    with np.load(tmp_path / "output.npz") as output:
        np.testing.assert_array_equal(output["depth"], expected)
    assert summary["output_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert summary["npu_calls"] == summary["submission_group_dispatches"] == 1
    for operation in ("pack", "unpack"):
        assert summary[f"codec_{operation}_ms_total"] == (
            summary[f"native_{operation}_ms"] + summary[f"vendor_{operation}_ms"])
        assert summary[f"native_{operation}_calls"] == (0 if mode == "vendor" else 1)
        assert summary[f"vendor_{operation}_calls"] == (1 if mode == "vendor" else 0)
    assert vendor.pack_calls == vendor.unpack_calls == (1 if mode == "vendor" else 0)
    if mode != "vendor":
        assert CpuDma.events[:4] == ["validate:input", "validate:output", "transport", "ensure_bank"]
