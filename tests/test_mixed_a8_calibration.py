import json

import numpy as np

from tools.calibrate_a8b8_linears_convs import (
    balanced_mse_statistics,
    load_input_records,
)


def test_input_list_preserves_domains_and_resolves_relative_paths(tmp_path):
    np.save(tmp_path / "nyu.npy", np.zeros((1, 3, 2, 2), dtype=np.float32))
    np.save(tmp_path / "da2k.npy", np.ones((1, 3, 2, 2), dtype=np.float32))
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"samples": [
        {"path": "nyu.npy", "domain": "nyu", "sample_id": "n0"},
        {"path": "da2k.npy", "domain": "da2k", "sample_id": "d0"},
    ]}))

    records = load_input_records([], manifest)

    assert [record["domain"] for record in records] == ["nyu", "da2k"]
    assert [record["sample_id"] for record in records] == ["n0", "d0"]
    assert all(record["path"].is_absolute() for record in records)


def test_balanced_mse_scale_weights_domains_equally():
    samples = {
        "nyu": [np.linspace(-1.0, 1.0, 512, dtype=np.float32) for _ in range(4)],
        "da2k": [np.linspace(-4.0, 4.0, 512, dtype=np.float32) for _ in range(4)],
    }

    result = balanced_mse_statistics(
        samples, {(1, 512)}, candidate_count=33, minimum_percentile=99.0
    )

    assert result["shape"] == [1, 512]
    assert result["domains"] == {"nyu": 4, "da2k": 4}
    assert result["a8_scale"] > 0.0
    selection = result["a8_scale_selection"]
    assert set(selection["best_validation"]) == {"nyu", "da2k"}
    selected = selection["selected"]
    expected = np.mean([
        selected["domains"][domain]["training"]["relative_l2"]
        for domain in ("nyu", "da2k")
    ])
    assert selected["balanced_training_relative_l2"] == expected
