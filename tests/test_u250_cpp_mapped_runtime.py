from __future__ import annotations

import unittest

import numpy as np

from tools.u250_cpp_mapped_runtime import (
    CppMappedRuntime, DDR_BASES, DeviceTensorHandle, merge_combined_ddr,
    record_span_per_bank, split_combined_ddr,
)
from tools.u250_layout_descriptors import TensorLayoutDescriptor


class FakeDmaBatch:
    def __init__(self):
        self.h2c = []
        self.c2h = []
        self.programs = []

    def h2c_batch_safe(self, requests):
        self.h2c.append(requests)

    def h2c_batch(self, requests):
        self.h2c.append(requests)

    def c2h_batch_safe(self, requests):
        self.c2h.append(requests)
        return [np.full(size, index, np.uint8)
                for index, (_, _, size) in enumerate(requests)]

    def c2h_batch(self, requests):
        return self.c2h_batch_safe(requests)

    def run_npu_chain(self, programs, timeout_ms):
        self.programs.append((programs, timeout_ms))
        return [0.001] * len(programs)

    def stats(self):
        return {"fake": True}

    def reset_stats(self):
        return None


class FakeExtension:
    DmaBatch = FakeDmaBatch


class FakeTransactionDmaBatch(FakeDmaBatch):
    def __init__(self):
        super().__init__()
        self.transactions = 0

    def run_resident_transaction(self, h2c, programs, c2h, timeout_ms, safe):
        self.transactions += 1
        self.h2c.append(h2c)
        self.programs.append((programs, timeout_ms))
        self.c2h.append(c2h)
        return {
            "outputs": [np.full(size, index, np.uint8)
                        for index, (_, _, size) in enumerate(c2h)],
            "npu_seconds": [0.001] * len(programs),
            "h2c_seconds": 0.002,
            "c2h_seconds": 0.003,
        }

    def run_decoder_capture_stems(self, stems, timeout_ms, safe):
        self.transactions += 1
        outputs = []
        programs = 0
        for stem in stems:
            raw = []
            for target in stem["targets"]:
                programs += 1
                raw.extend(np.full(size, index, np.uint8)
                           for index, (_, _, size) in enumerate(
                               target["output_requests"]
                           ))
            outputs.append(raw)
        return {
            "outputs": outputs,
            "npu_seconds": [0.001] * programs,
            "source_c2h_seconds": 0.001,
            "h2c_seconds": 0.002,
            "c2h_seconds": 0.004,
            "bridge_seconds": 0.003,
            "programs": programs,
        }

    def run_frame_graph(self, initial_tensors, nodes, fetches, timeout_ms, safe):
        self.transactions += 1
        tensors = dict(initial_tensors)
        timings = []
        opcode_seconds = {}
        node_seconds = []
        programs = 0
        for node in nodes:
            op = node["op"]
            elapsed = 0.001
            if op == "device_read":
                self.c2h.append([
                    request for group in node["requests"] for request in group
                ])
                for name, requests in zip(node["outputs"], node["requests"]):
                    tensors[name] = tuple(
                        np.full(size, bank, np.uint8)
                        for bank, (_, _, size) in enumerate(requests)
                    )
            elif op == "decoder_capture_pack_bf16":
                half = node["target_descriptor"]["combined_bytes"] // 2
                tensors[node["output"]] = tuple(
                    np.zeros(half, np.uint8) for _ in range(2)
                )
            elif op == "device_write":
                self.h2c.append([
                    (bank, addresses[bank], tensors[name][bank])
                    for name, addresses in zip(node["inputs"], node["addresses"])
                    for bank in range(2)
                ])
            elif op == "npu_chain":
                self.programs.append((node["programs"], timeout_ms))
                programs += len(node["programs"])
                timings.extend([0.001] * len(node["programs"]))
            opcode_seconds[op] = opcode_seconds.get(op, 0.0) + elapsed
            node_seconds.append(elapsed)
        return {
            "outputs": {name: tensors[name] for name in fetches},
            "npu_seconds": timings,
            "node_seconds": node_seconds,
            "opcode_seconds": opcode_seconds,
            "nodes": len(nodes),
            "programs": programs,
            "peak_tensors": len(tensors),
            "wall_seconds": sum(node_seconds),
        }


class FakeTransactionExtension:
    DmaBatch = FakeTransactionDmaBatch


class ResidentSelection:
    mode = "vendor"
    extension_path = None

    def __init__(self, descriptors):
        self.descriptors = descriptors

    def prepare(self, codec_type):
        return None

    def native_for(self, name, direction):
        return True

    def stats(self):
        return {}

    def reset_stats(self):
        return None


def record(name="case"):
    return {
        "name": name,
        "base_addresses": [10, 20, 30, 40, 1000, 50],
        "isa_ranges": [2, 1],
        "inputs": [{"address": 0, "size_per_bank": 512}],
        "outputs": [{"address": 4, "size_per_bank": 512}],
    }


def descriptor(direction, index=0, *, bitdepth=16, combined_bytes=512):
    return TensorLayoutDescriptor(
        layout="NDWC", dims=(1, 1, 16, 16), bitdepth=bitdepth,
        c_align=2 if bitdepth == 16 else 1,
        w_align=2 if bitdepth == 16 else 1,
        combined_bytes=combined_bytes, direction=direction, index=index,
        matrix_role="left" if direction == "input" else "output",
    )


def nchw_descriptor(direction, index=0, *, channels=16, height=1, width=16):
    return TensorLayoutDescriptor(
        layout="NCHW", dims=(1, channels, height, width), bitdepth=8,
        c_align=(channels + 15) // 16,
        w_align=((width + 15) // 16) * ((channels + 15) // 16),
        combined_bytes=(height * ((width + 15) // 16)
                        * ((channels + 15) // 16) * 256),
        direction=direction, index=index, matrix_role="netio",
    )


def resident_records():
    norm = {
        "name": "norm", "base_addresses": [10, 20, 30, 40, 1000, 50],
        "isa_ranges": [2, 1],
        "inputs": [{"address": 0, "size_per_bank": 512}],
        "outputs": [{"address": 4, "size_per_bank": 512}],
    }
    post = {
        "name": "post", "base_addresses": [11, 21, 31, 41, 1000, 51],
        "isa_ranges": [3, 1],
        "inputs": [
            {"address": 0, "size_per_bank": 256},
            {"address": 2, "size_per_bank": 512},
        ],
        "outputs": [{"address": 6, "size_per_bank": 512}],
    }
    descriptors = {
        "norm": {"input": [descriptor("input")],
                 "output": [descriptor("output")]},
        "post": {"input": [descriptor("input", 0, bitdepth=8,
                                             combined_bytes=256),
                            descriptor("input", 1)],
                 "output": [descriptor("output")]},
    }
    return norm, post, descriptors


class CppMappedRuntimeTest(unittest.TestCase):
    def test_split_merge_round_trip(self):
        value = np.arange(1024, dtype=np.uint8)
        even, odd = split_combined_ddr(value)
        self.assertEqual(even.size, 512)
        self.assertTrue(np.array_equal(merge_combined_ddr(even, odd), value))

    def test_group_uses_disjoint_slots_and_exact_c2h_halves(self):
        manifest = {"shared_fm_workspace_bytes": 32768}
        runtime = CppMappedRuntime(manifest, FakeExtension)
        packed = np.arange(512, dtype=np.uint8)
        outputs, timing = runtime.run_group(
            [record("a"), record("b")], [[packed], [packed]],
            [[True], [True]], 1234,
        )
        stride = record_span_per_bank(record())
        programs, timeout = runtime.transport.programs[0]
        self.assertEqual(timeout, 1234)
        self.assertEqual(programs[1]["base_addresses"][4], 1000 + stride // 128)
        requests = runtime.transport.c2h[0]
        self.assertEqual([item[2] for item in requests], [256, 256, 256, 256])
        self.assertEqual(requests[2][1] - requests[0][1], stride)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0][0].size, 512)
        self.assertEqual(timing["c2h_bytes"], 1024)
        self.assertEqual(timing["submission_group_size"], 2)

    def test_group_compiles_to_one_cpp_frame_graph_when_available(self):
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 32768}, FakeTransactionExtension
        )
        packed = np.arange(512, dtype=np.uint8)
        outputs, timing = runtime.run_group(
            [record("a"), record("b")], [[packed], [packed]],
            [[True], [True]], 1234,
        )
        self.assertEqual(runtime.transport.transactions, 1)
        self.assertEqual(len(runtime.transport.h2c), 1)
        self.assertEqual(len(runtime.transport.programs), 1)
        self.assertEqual(len(runtime.transport.c2h), 1)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(timing["cpp_resident_transaction"])
        self.assertEqual(timing["h2c_ms"], 1.0)
        self.assertEqual(timing["c2h_ms"], 1.0)
        stats = runtime.stats()
        self.assertEqual(stats["python_transport_api_calls"], 1)
        self.assertEqual(stats["cpp_resident_transaction_calls"], 1)
        self.assertEqual(stats["frame_graph_nodes"], 3)

    def test_decoder_capture_stems_use_one_cpp_frame_graph_call(self):
        project = {
            "name": "project", "base_addresses": [10, 20, 30, 40, 1000, 50],
            "isa_ranges": [2, 1],
            "inputs": [{"address": 0, "size_per_bank": 256}],
            "outputs": [{"address": 2, "size_per_bank": 512}],
        }
        source = descriptor("output")
        target = nchw_descriptor("input")
        selection = ResidentSelection({
            "project": {"input": [target], "output": [descriptor("output")]}
        })
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 32768}, FakeTransactionExtension,
            codec_selection=selection,
        )
        physical_source = tuple(np.zeros(256, np.uint8) for _ in range(2))
        outputs, timing = runtime.run_decoder_capture_stems([{
            "source": physical_source,
            "source_descriptor": source,
            "target_descriptor": target,
            "gamma": np.ones(16, np.float32),
            "beta": np.zeros(16, np.float32),
            "scale": 0.125,
            "epsilon": 1.0e-6,
            "records": [project],
        }], 1000)
        self.assertEqual(runtime.transport.transactions, 1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0]), 1)
        self.assertEqual(len(outputs[0][0]), 1)
        self.assertTrue(timing["cpp_frame_graph"])
        self.assertEqual(timing["submission_groups"], 1)
        self.assertEqual(timing["submission_group_size"], 1)
        self.assertEqual(runtime.stats()["python_transport_api_calls"], 1)

    def test_bank_load_is_two_parallel_requests(self):
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 32768}, FakeExtension
        )
        runtime.load_bank(np.arange(1024, dtype=np.uint8))
        requests = runtime.transport.h2c[0]
        self.assertEqual([item[0] for item in requests], [0, 1])
        self.assertEqual([item[1] for item in requests], list(DDR_BASES))
        self.assertEqual([item[2].size for item in requests], [512, 512])

    def test_resident_bank_is_loaded_once(self):
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 32768}, FakeExtension
        )
        value = np.arange(1024, dtype=np.uint8)
        first_ms, first_reused = runtime.ensure_bank(value, "same")
        second_ms, second_reused = runtime.ensure_bank(value, "same")
        self.assertGreaterEqual(first_ms, 0.0)
        self.assertFalse(first_reused)
        self.assertEqual(second_ms, 0.0)
        self.assertTrue(second_reused)
        self.assertEqual(len(runtime.transport.h2c), 1)
        with self.assertRaises(RuntimeError):
            runtime.ensure_bank(value, "different")

    def test_vendor_stats_expose_zero_native_counters_without_codec_api(self):
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 32768}, FakeExtension
        )
        runtime.run_group([record()], [[np.zeros(512, np.uint8)]], [[True]], 100)
        stats = runtime.stats()
        self.assertEqual(stats["layout_codec"], "vendor")
        self.assertEqual(stats["native_pack_calls"], 0)
        self.assertEqual(stats["native_unpack_calls"], 0)
        self.assertEqual(stats["physical_npu_dispatches"], 1)
        self.assertEqual(stats["fallback_reasons"], {})

    def test_resident_handle_forwards_input_and_connects_output(self):
        norm, post, descriptors = resident_records()
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 65536}, FakeExtension,
            codec_selection=ResidentSelection(descriptors),
        )
        bf16 = tuple(np.zeros(256, np.uint8) for _ in range(2))
        int8 = tuple(np.zeros(128, np.uint8) for _ in range(2))
        _, norm_inputs, _, _ = runtime.run_resident_chain(
            [norm], [[bf16]], [10], {}, 1000,
        )
        x_handle = norm_inputs[0][0]
        self.assertIsInstance(x_handle, DeviceTensorHandle)

        outputs, inputs, handles, timing = runtime.run_resident_chain(
            [post, norm], [[int8, x_handle], [None]], [8, 14],
            {(1, 0): (0, 0)}, 1000,
        )
        programs, _ = runtime.transport.programs[-1]
        self.assertEqual([item["base_addresses"][4] for item in programs],
                         [1008, 1014])
        self.assertIs(inputs[0][1], x_handle)
        self.assertEqual(
            handles[0][0].bank_addresses,
            ((1000 + 14) * 128 + DDR_BASES[0],
             (1000 + 14) * 128 + DDR_BASES[1]),
        )
        self.assertEqual(len(outputs), 2)
        self.assertEqual(timing["h2c_bytes"], 256)
        self.assertEqual(timing["h2c_skipped_bytes"], 1024)
        self.assertEqual(timing["resident_forwarded_inputs"], 1)
        self.assertEqual(timing["resident_connections"], 1)

    def test_overlapping_group_write_invalidates_device_handle(self):
        norm, _, descriptors = resident_records()
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 65536}, FakeExtension,
            codec_selection=ResidentSelection(descriptors),
        )
        bf16 = tuple(np.zeros(256, np.uint8) for _ in range(2))
        _, inputs, _, _ = runtime.run_resident_chain(
            [norm], [[bf16]], [0], {}, 1000,
        )
        handle = inputs[0][0]
        runtime.run_group([norm], [[bf16]], [[True]], 1000)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            runtime.run_resident_chain([norm], [[handle]], [0], {}, 1000)

    def test_frame_reset_expires_device_handle_before_dma(self):
        norm, _, descriptors = resident_records()
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 65536}, FakeExtension,
            codec_selection=ResidentSelection(descriptors),
        )
        bf16 = tuple(np.zeros(256, np.uint8) for _ in range(2))
        _, inputs, _, _ = runtime.run_resident_chain(
            [norm], [[bf16]], [0], {}, 1000,
        )
        handle = inputs[0][0]
        h2c_batches = len(runtime.transport.h2c)
        runtime.reset_frame_stats()
        with self.assertRaisesRegex(RuntimeError, "expired frame lifetime"):
            runtime.run_resident_chain([norm], [[handle]], [0], {}, 1000)
        self.assertEqual(len(runtime.transport.h2c), h2c_batches)

    def test_resident_connection_rejects_address_and_abi_mismatch(self):
        norm, post, descriptors = resident_records()
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 65536}, FakeExtension,
            codec_selection=ResidentSelection(descriptors),
        )
        int8 = tuple(np.zeros(128, np.uint8) for _ in range(2))
        bf16 = tuple(np.zeros(256, np.uint8) for _ in range(2))
        with self.assertRaisesRegex(RuntimeError, "address mismatch"):
            runtime.run_resident_chain(
                [post, norm], [[int8, bf16], [None]], [8, 13],
                {(1, 0): (0, 0)}, 1000,
            )
        changed = dict(descriptors)
        changed["norm"] = dict(descriptors["norm"])
        changed["norm"]["input"] = [
            descriptor("input", bitdepth=8, combined_bytes=256)
        ]
        runtime.codec_selection = ResidentSelection(changed)
        with self.assertRaisesRegex(RuntimeError, "storage ABI mismatch"):
            runtime.run_resident_chain(
                [post, norm], [[int8, bf16], [None]], [8, 14],
                {(1, 0): (0, 0)}, 1000,
            )

    def test_earlier_output_cannot_silently_replace_later_preloaded_input(self):
        norm, _, descriptors = resident_records()
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 65536}, FakeExtension,
            codec_selection=ResidentSelection(descriptors),
        )
        bf16 = tuple(np.zeros(256, np.uint8) for _ in range(2))
        with self.assertRaisesRegex(RuntimeError, "overwrites preloaded input"):
            runtime.run_resident_chain(
                [norm, norm], [[bf16], [bf16]], [0, 4], {}, 1000,
            )

    def test_transport_failure_invalidates_all_device_handles(self):
        norm, _, descriptors = resident_records()
        runtime = CppMappedRuntime(
            {"shared_fm_workspace_bytes": 65536}, FakeExtension,
            codec_selection=ResidentSelection(descriptors),
        )
        bf16 = tuple(np.zeros(256, np.uint8) for _ in range(2))

        def fail(_programs, _timeout_ms):
            raise TimeoutError("injected NPU timeout")

        runtime.transport.run_npu_chain = fail
        with self.assertRaisesRegex(TimeoutError, "injected NPU timeout"):
            runtime.run_resident_chain([norm], [[bf16]], [0], {}, 1000)
        stats = runtime.stats()
        self.assertEqual(stats["device_tensor_handle_creations"], 1)
        self.assertEqual(stats["device_tensor_handle_invalidations"], 1)
        self.assertEqual(stats["device_tensor_live_handles"], 0)


if __name__ == "__main__":
    unittest.main()
