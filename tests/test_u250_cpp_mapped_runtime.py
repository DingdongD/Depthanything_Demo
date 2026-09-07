from __future__ import annotations

import unittest

import numpy as np

from tools.u250_cpp_mapped_runtime import (
    CppMappedRuntime, DDR_BASES, merge_combined_ddr, record_span_per_bank,
    split_combined_ddr,
)


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


def record(name="case"):
    return {
        "name": name,
        "base_addresses": [10, 20, 30, 40, 1000, 50],
        "isa_ranges": [2, 1],
        "inputs": [{"address": 0, "size_per_bank": 512}],
        "outputs": [{"address": 4, "size_per_bank": 512}],
    }


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


if __name__ == "__main__":
    unittest.main()
