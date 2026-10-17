"""Tests for the train->live-module guards and the adapter verifier.

Torch/peft are optional: the tensor-level guards and the safetensors verifier are exercised
with lightweight fakes and a hand-written safetensors file, so the suite runs with numpy
alone. Torch-only paths are skipped when torch is absent.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import guards
import verify_adapter_is_live as V


class _FakeParam:
    """Stand-in for a torch parameter exposing .detach().float().norm().item() and .requires_grad."""

    def __init__(self, l2: float, requires_grad: bool = True):
        self._l2 = float(l2)
        self.requires_grad = requires_grad

    # detach()->self, float()->self, norm()->self, item()->value
    def detach(self):
        return self

    def float(self):
        return self

    def norm(self):
        return self

    def item(self):
        return self._l2


class _FakeModel:
    def __init__(self, named):
        self._named = named  # list[(name, _FakeParam)]

    def named_parameters(self):
        return list(self._named)


class _FakeOptimizer:
    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]


def _trained_model():
    a = _FakeParam(0.4, True)
    b = _FakeParam(0.7, True)
    return _FakeModel([
        ("base.layers.0.q_proj.lora_A.weight", a),
        ("base.layers.0.q_proj.lora_B.weight", b),
    ]), a, b


def _init_model():
    a = _FakeParam(0.4, True)
    b = _FakeParam(0.0, True)  # B at init
    return _FakeModel([
        ("base.layers.0.q_proj.lora_A.weight", a),
        ("base.layers.0.q_proj.lora_B.weight", b),
    ]), a, b


class TestLoraTrainedGuard(unittest.TestCase):
    def test_trained_passes(self):
        m, _a, _b = _trained_model()
        self.assertAlmostEqual(guards.assert_lora_trained(m), 0.7)

    def test_init_fails(self):
        m, _a, _b = _init_model()
        with self.assertRaises(guards.AdapterNotTrainedError):
            guards.assert_lora_trained(m)

    def test_no_lora_fails(self):
        m = _FakeModel([("base.layers.0.q_proj.weight", _FakeParam(1.0, True))])
        with self.assertRaises(guards.AdapterNotTrainedError):
            guards.assert_lora_trained(m)


class TestOptimizerTracksModel(unittest.TestCase):
    def test_tracked_passes(self):
        m, a, b = _trained_model()
        opt = _FakeOptimizer([a, b])
        self.assertEqual(guards.assert_optimizer_tracks_model(opt, m), 2)

    def test_decoupled_fails(self):
        m, _a, _b = _trained_model()
        opt = _FakeOptimizer([_FakeParam(0.1, True)])  # different objects entirely
        with self.assertRaises(guards.OptimizerDecoupledError):
            guards.assert_optimizer_tracks_model(opt, m)

    def test_no_trainable_lora_fails(self):
        frozen = _FakeModel([("base.layers.0.q_proj.lora_B.weight", _FakeParam(0.0, False))])
        opt = _FakeOptimizer([])
        with self.assertRaises(guards.OptimizerDecoupledError):
            guards.assert_optimizer_tracks_model(opt, frozen)


class TestResponsesNotIdentical(unittest.TestCase):
    def test_identical_fails(self):
        with TemporaryDirectory() as d:
            a = Path(d) / "a.jsonl"
            b = Path(d) / "b.jsonl"
            a.write_text('{"id": 1, "choice": "loyal"}\n')
            b.write_text('{"id": 1, "choice": "loyal"}\n')
            with self.assertRaises(guards.IdenticalArmsError):
                guards.assert_responses_not_identical(a, b)

    def test_distinct_passes(self):
        with TemporaryDirectory() as d:
            a = Path(d) / "a.jsonl"
            b = Path(d) / "b.jsonl"
            a.write_text('{"id": 1, "choice": "loyal"}\n')
            b.write_text('{"id": 1, "choice": "neutral"}\n')
            guards.assert_responses_not_identical(a, b)  # no raise

    def test_all_arms_distinct(self):
        with TemporaryDirectory() as d:
            paths = []
            for i, txt in enumerate(["x", "y", "x"]):
                p = Path(d) / f"{i}.jsonl"
                p.write_text(txt)
                paths.append(p)
            with self.assertRaises(guards.IdenticalArmsError):
                guards.assert_all_arms_distinct(paths)


def _write_safetensors(path: Path, tensors: dict[str, list[float]]) -> None:
    """Write a minimal float32 safetensors file for {name: flat float list}."""
    import numpy as np

    header = {}
    blob = b""
    offset = 0
    for name, values in tensors.items():
        arr = np.asarray(values, dtype="<f4")
        data = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": [len(values)],
                        "data_offsets": [offset, offset + len(data)]}
        blob += data
        offset += len(data)
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        fh.write(blob)


class TestVerifier(unittest.TestCase):
    def test_trained_adapter_ok(self):
        with TemporaryDirectory() as d:
            fp = Path(d) / "adapter_model.safetensors"
            _write_safetensors(fp, {
                "m.q_proj.lora_A.weight": [0.1, 0.2, -0.3, 0.05],
                "m.q_proj.lora_B.weight": [0.4, -0.6, 0.2, 0.1],
            })
            v = V.summarize(Path(d))
            self.assertTrue(v["ok"], v)
            self.assertGreater(v["b_norm_max"], 1e-6)

    def test_inert_adapter_flagged(self):
        with TemporaryDirectory() as d:
            fp = Path(d) / "adapter_model.safetensors"
            _write_safetensors(fp, {
                "m.q_proj.lora_A.weight": [0.1, 0.2, -0.3, 0.05],
                "m.q_proj.lora_B.weight": [0.0, 0.0, 0.0, 0.0],  # B at init
            })
            v = V.summarize(Path(d))
            self.assertFalse(v["ok"], v)
            self.assertIn("initialisation", v.get("error", ""))

    def test_main_exit_codes(self):
        with TemporaryDirectory() as d:
            good = Path(d) / "good"
            good.mkdir()
            _write_safetensors(good / "adapter_model.safetensors", {
                "m.lora_A.weight": [0.1, 0.2], "m.lora_B.weight": [0.3, 0.4]})
            bad = Path(d) / "bad"
            bad.mkdir()
            _write_safetensors(bad / "adapter_model.safetensors", {
                "m.lora_A.weight": [0.1, 0.2], "m.lora_B.weight": [0.0, 0.0]})
            self.assertEqual(V.main([str(good)]), 0)
            self.assertEqual(V.main([str(bad)]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
