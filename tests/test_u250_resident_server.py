import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools" / "depthanything_u250_resident_server.py"


def test_resident_server_preserves_request_identity_and_reuses_runner(tmp_path):
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        """
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

calls = 0

def main():
    global calls
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--golden', type=Path)
    args = parser.parse_args()
    calls += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = np.array([calls], dtype=np.float32)
    np.savez(args.output, depth=value)
    digest = hashlib.sha256(value.tobytes()).hexdigest()
    args.output.with_suffix('.summary.json').write_text(json.dumps({
        'output_sha256': digest,
        'wall_ms': 10.0 + calls,
        'process_wall_ms': 20.0 + calls,
        'resident_bank_reused': calls > 1,
        'cpp_runtime_reused': calls > 1,
    }))
    return 0
"""
    )
    base_args = tmp_path / "base_args.json"
    base_args.write_text("[]\n")
    requests = [
        {"id": "warmup:sample", "input": "first.npy",
         "output": str(tmp_path / "warmup.npz")},
        {"id": "split/sample", "input": "second.npy",
         "output": str(tmp_path / "sample.npz")},
        {"command": "shutdown"},
    ]
    result = subprocess.run(
        [sys.executable, str(SERVER), "--runner", str(runner),
         "--base-args", str(base_args)],
        input="\n".join(json.dumps(item) for item in requests) + "\n",
        text=True, capture_output=True, check=True,
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[0]["ready"] is True
    assert events[1]["request_id"] == "warmup:sample"
    assert events[1]["resident_bank_reused"] is False
    assert events[2]["request_id"] == "split/sample"
    assert events[2]["cpp_runtime_reused"] is True
    assert events[2]["runtime_wall_ms"] == 12.0
    assert events[2]["process_wall_ms"] == 22.0
    assert events[3] == {"shutdown": True}
