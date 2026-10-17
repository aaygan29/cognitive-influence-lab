"""Smoke tests for the corrected trainer's save->verify gate (no torch/GPU).

The training loop itself needs a model, but the load-bearing FIX is the guard at save time: a
saved adapter that is at initialisation must abort rather than become a base-model null. That is
tested here with a fake model whose save_pretrained writes a chosen adapter.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import guards
import reference_trainer as RT


def _write_safetensors(path: Path, tensors: dict[str, list[float]]) -> None:
    import numpy as np

    header, blob, offset = {}, b"", 0
    for name, values in tensors.items():
        data = np.asarray(values, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": [len(values)],
                        "data_offsets": [offset, offset + len(data)]}
        blob += data
        offset += len(data)
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        fh.write(blob)


class _FakeModel:
    """Writes a prescribed adapter file on save_pretrained; nothing else."""

    def __init__(self, tensors):
        self._tensors = tensors

    def save_pretrained(self, path):
        _write_safetensors(Path(path) / "adapter_model.safetensors", self._tensors)


class TestSaveGate(unittest.TestCase):
    def test_inert_adapter_aborts_save(self):
        t = RT.Trainer("fake-model")
        t.model = _FakeModel({"m.lora_A.weight": [0.1, 0.2], "m.lora_B.weight": [0.0, 0.0]})
        with TemporaryDirectory() as d:
            with self.assertRaises(guards.AdapterNotTrainedError):
                t.save(Path(d) / "organism")

    def test_trained_adapter_saves(self):
        t = RT.Trainer("fake-model")
        t.model = _FakeModel({"m.lora_A.weight": [0.1, 0.2], "m.lora_B.weight": [0.3, 0.4]})
        with TemporaryDirectory() as d:
            out = t.save(Path(d) / "organism")   # no _verify_probe set -> skips reload step
            self.assertTrue((Path(out) / "adapter_model.safetensors").exists())

    def test_imports_and_constructs(self):
        t = RT.Trainer("Qwen/Qwen2.5-7B-Instruct", lora_r=16, lora_alpha=32, seed=20260930)
        self.assertEqual(t.lora_r, 16)
        self.assertIsNone(t.model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
