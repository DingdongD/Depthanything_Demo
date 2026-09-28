from tools.enable_u250_decoder_layernorm_contract import lower_decoder_layernorm


def contract(norms=4):
    nodes = []
    for index in range(norms):
        nodes.append({
            "name": f"norm{index}", "op_type": "LayerNormalization",
            "inputs": [f"capture{index}", "gamma", "beta"],
            "outputs": [f"normalized{index}"],
        })
    nodes.append({"name": "slice", "op_type": "Slice", "inputs": [], "outputs": []})
    return {
        "decoder": [{"backend": "host", "nodes": nodes}],
        "execution_totals": {"decoder_npu_calls": 89, "npu_calls_per_inference": 443},
    }


def test_lowering_preserves_host_tail_and_updates_exact_call_totals():
    result = lower_decoder_layernorm(contract(), "tail_norm")
    assert [step["backend"] for step in result["decoder"][:4]] == [
        "npu_layernorm"] * 4
    assert result["decoder"][4]["nodes"][0]["name"] == "slice"
    assert result["execution_totals"]["decoder_npu_calls"] == 93
    assert result["execution_totals"]["npu_calls_per_inference"] == 447


def test_lowering_rejects_incomplete_decoder_norm_coverage():
    try:
        lower_decoder_layernorm(contract(3), "tail_norm")
    except ValueError as error:
        assert "expected four" in str(error)
    else:
        raise AssertionError("incomplete decoder LayerNorm coverage was accepted")
