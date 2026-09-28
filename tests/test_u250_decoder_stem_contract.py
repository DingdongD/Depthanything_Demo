from tools.enable_u250_decoder_stem_contract import enable_stems


def test_four_project_convs_are_replaced_without_changing_dispatch_count():
    decoder = []
    bank_cases = []
    kernels = []
    for project, channels in enumerate((48, 96, 192, 384)):
        pieces = []
        for begin in range(0, channels, 64):
            end = min(begin + 64, channels)
            name = f"decoder_stem_{project}_co{begin:03d}_{end:03d}"
            pieces.append({
                "name": name, "project_index": project,
                "channel_start": begin, "channel_end": end,
                "output_shape": [1, end - begin, 37, 37],
            })
            bank_cases.append({"name": name})
        kernels.extend(pieces)
        decoder.append({
            "backend": "npu", "source_node":
            f"/depth_head/projects.{project}/Conv",
            "input_tensor": "old", "output_tensor": f"project_{project}",
            "kernels": [{"name": f"old_{project}"}],
        })
    contract = {
        "decoder": decoder,
        "execution_totals": {"decoder_npu_calls": 12,
                             "npu_calls_per_inference": 443},
    }
    manifest = {"affine_mode": "epu", "kernels": kernels}
    bank = {
        "cases": bank_cases, "bank_size_bytes": 4096, "bank_sha256": "digest",
        "shared_fm_workspace_bytes": 16_777_216,
        "required_fm_io_bytes": 6_285_312,
    }
    result = enable_stems(contract, manifest, bank)
    assert len(result["decoder"]) == 4
    assert all(step["fused_decoder_stem"] for step in result["decoder"])
    assert sum(len(step["kernels"]) for step in result["decoder"]) == 12
    assert result["execution_totals"]["decoder_npu_calls"] == 12
    assert result["execution_totals"]["npu_calls_per_inference"] == 443
    assert result["decoder"][3]["capture_layer"] == 11


def test_partial_fusion_preserves_unqualified_project():
    decoder = [{
        "backend": "npu", "source_node": f"/depth_head/projects.{project}/Conv",
        "input_tensor": "old", "output_tensor": f"project_{project}",
        "kernels": [{"name": f"old_{project}"}],
    } for project in range(4)]
    kernels = [{
        "name": f"stem_{project}", "project_index": project,
        "channel_start": 0, "channel_end": 1, "output_shape": [1, 1, 37, 37],
    } for project in range(4)]
    bank = {
        "cases": [{"name": item["name"]} for item in kernels],
        "bank_size_bytes": 4096, "bank_sha256": "digest",
        "shared_fm_workspace_bytes": 16_777_216,
        "required_fm_io_bytes": 6_285_312,
    }
    contract = {"decoder": decoder, "execution_totals": {}}
    result = enable_stems(
        contract, {"affine_mode": "fold", "kernels": kernels}, bank,
        (0, 1, 2),
    )
    assert [step.get("fused_decoder_stem", False) for step in result["decoder"]] == [
        True, True, True, False,
    ]
    assert result["decoder_capture_policy"]["layers"] == [2, 5, 8]


def test_patch_input_keeps_slice_on_host_and_fuses_only_layout_project():
    contract = {
        "decoder": [{
            "backend": "npu",
            "source_node": "/depth_head/projects.0/Conv",
            "input_tensor": "old",
            "output_tensor": "project_0",
            "kernels": [{"name": "old_0"}],
        }],
        "execution_totals": {},
    }
    kernel = {
        "name": "decoder_patch_project_0_co000_048",
        "project_index": 0,
        "channel_start": 0,
        "channel_end": 48,
        "input_tensor": "/Slice_1_output_0",
        "input_shape": [1, 1369, 384],
        "output_shape": [1, 48, 37, 37],
    }
    bank = {
        "cases": [{"name": kernel["name"]}],
        "bank_size_bytes": 4096,
        "bank_sha256": "digest",
        "shared_fm_workspace_bytes": 16_777_216,
        "required_fm_io_bytes": 6_285_312,
    }
    result = enable_stems(
        contract,
        {"affine_mode": "fold", "stem_input": "patches", "kernels": [kernel]},
        bank,
        (0,),
    )
    step = result["decoder"][0]
    assert step["input_tensor"] == "/Slice_1_output_0"
    assert step["input_shape"] == [1, 1369, 384]
    assert "/Slice_1" not in step["fused_host_nodes"]
    assert step["fused_host_nodes"] == [
        "/depth_head/Transpose",
        "/depth_head/Constant",
        "/depth_head/Reshape",
    ]
    assert result["decoder_stem_policy"]["fused_ops"] == [
        "Transpose", "Reshape", "Conv",
    ]
    assert "decoder_capture_policy" not in result
