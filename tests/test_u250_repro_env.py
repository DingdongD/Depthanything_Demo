from argparse import Namespace
from pathlib import Path

from tools.check_u250_repro_env import (
    inspect_environment,
    path_check,
    runtime_extension_check,
)


def make_args(tmp_path: Path) -> Namespace:
    toolchain = tmp_path / "toolchain"
    (toolchain / "python_bin").mkdir(parents=True)
    (toolchain / "python_libs").mkdir()
    (toolchain / "python_bin/compile.py").write_text("# fixture\n")
    (toolchain / "arch_16_mono.yaml").write_text("fixture: true\n")
    (toolchain / "arch_256_mono.yaml").write_text("fixture: true\n")
    python_root = tmp_path / "acmlir"
    extension = python_root / "RelWithDebInfo/lib"
    extension.mkdir(parents=True)
    checkpoint = tmp_path / "depth_anything_v2_vits.pth"
    checkpoint.write_bytes(b"fixture")
    return Namespace(
        checkpoint=checkpoint,
        toolchain_root=toolchain,
        compiler=None,
        arch_16=None,
        arch_256=None,
        python_root=python_root,
        extension_dir=None,
        compiler_python="python",
        runtime_dir=None,
        require_board=False,
        skip_python_packages=True,
    )


def test_repro_doctor_accepts_complete_external_contract(tmp_path):
    report = inspect_environment(make_args(tmp_path))
    assert report["ok"] is True
    assert all(item["ok"] for item in report["checks"])


def test_repro_doctor_reports_missing_checkpoint(tmp_path):
    args = make_args(tmp_path)
    args.checkpoint.unlink()
    report = inspect_environment(args)
    assert report["ok"] is False
    checkpoint = next(item for item in report["checks"]
                      if item["name"] == "checkpoint")
    assert checkpoint["ok"] is False


def test_repro_doctor_accepts_character_device_paths():
    assert path_check("null", Path("/dev/null"), "path")["ok"] is True


def test_runtime_extension_check_requires_exactly_one_abi(tmp_path):
    assert runtime_extension_check(tmp_path)["ok"] is False
    (tmp_path / "npz2bin.cpython-310-x86_64-linux-gnu.so").write_bytes(b"fixture")
    assert runtime_extension_check(tmp_path)["ok"] is True
    (tmp_path / "npz2bin.cpython-39-x86_64-linux-gnu.so").write_bytes(b"fixture")
    assert runtime_extension_check(tmp_path)["ok"] is False
