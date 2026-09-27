"""Guards against the train->live-module hand-off failure that produced a base-model null.

Background. In the "Auditing Rare-Condition Loyalties" run, training loss moved (organism
3.712 -> 3.462) yet the saved/evaluated LoRA adapter carried an *initialisation*: LoRA A at
its kaiming-uniform init and LoRA B ~ 0, so the adapter delta B*A ~ 0 and the model was the
base model. Both arms therefore emitted byte-identical responses (md5 d84b27bd...), and the
"measured null" was an artifact of the pipeline, not a fact about the loyalty construction.

These guards make that failure loud instead of silent. Each is a hard check meant to be
called at a specific boundary of a LoRA/PEFT training + evaluation pipeline:

  1. assert_optimizer_tracks_model(opt, model)   -- right after the optimizer is built
  2. assert_lora_trained(model)                  -- after the last opt.step(), before saving
  3. assert_adapter_changes_output(model, ids)   -- after the adapter is attached at eval
  4. assert_responses_not_identical(a, b)        -- before computing any arm-vs-arm contrast

They are framework-agnostic and import torch/peft lazily, so importing this module costs
nothing and the pure-file checks (4, and the norm helpers) run with no torch installed.

Design note: a null is only meaningful once (2) and (3) have both passed for the arm that is
supposed to carry the behaviour. A pipeline that reports a contrast without them cannot tell
"the construction did nothing" apart from "the trained weights never reached the model".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


class GuardError(AssertionError):
    """Base class for a guard that fired. Subclasses name the specific failure."""


class OptimizerDecoupledError(GuardError):
    """The optimizer does not step the model's trainable LoRA tensors (root cause 1)."""


class AdapterNotTrainedError(GuardError):
    """The live/saved LoRA adapter is at initialisation: B ~ 0, so it is inert."""


class AdapterInertError(GuardError):
    """An attached adapter does not change the model's output versus the base model."""


class IdenticalArmsError(GuardError):
    """Two arms that should differ produced byte-identical response files."""


# --------------------------------------------------------------------------------------
# LoRA tensor helpers (operate on a live peft/torch model; torch imported lazily)
# --------------------------------------------------------------------------------------

def _is_lora_b(name: str) -> bool:
    n = name.lower()
    return "lora_b" in n or "lora_embedding_b" in n


def _is_lora_a(name: str) -> bool:
    n = name.lower()
    return "lora_a" in n or "lora_embedding_a" in n


def lora_b_norms(model) -> dict[str, float]:
    """L2 norm of every LoRA B tensor in a live model, keyed by parameter name.

    LoRA B is initialised to exactly zero, so its norm is the cleanest single signal of
    whether the adapter learned anything: a trained adapter has ||B|| > 0, an untrained or
    decoupled one has ||B|| == 0 (or ~1e-8 numerical dust).
    """
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if _is_lora_b(name):
            out[name] = float(p.detach().float().norm().item())
    return out


def trained_lora_param_ids(model) -> dict[int, str]:
    """id(tensor) -> name for every trainable LoRA parameter on the model."""
    return {
        id(p): name
        for name, p in model.named_parameters()
        if p.requires_grad and "lora_" in name.lower()
    }


# --------------------------------------------------------------------------------------
# Guard 1: the optimizer must step the tensors that will be saved
# --------------------------------------------------------------------------------------

def assert_optimizer_tracks_model(optimizer, model) -> int:
    """Fail unless every trainable LoRA tensor on `model` is in `optimizer`'s param groups.

    Call this immediately after building the optimizer AND again after any wrapping step
    (accelerator.prepare, DDP, torch.compile). The classic failure is: build the optimizer,
    then replace `model` with a wrapped copy whose parameters the optimizer never sees. The
    copy learns (loss falls) while the module you save stays at init. Comparing tensor
    identities catches exactly that divergence.

    Returns the number of tracked LoRA tensors on success.
    """
    trained = trained_lora_param_ids(model)
    if not trained:
        raise OptimizerDecoupledError(
            "model has no trainable LoRA parameters (requires_grad + 'lora_' in name). "
            "Either get_peft_model was not applied, the target_modules missed every trained "
            "layer, or the adapter params were frozen before the optimizer was built."
        )
    opt_ids: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_ids.add(id(p))
    missing = [name for pid, name in trained.items() if pid not in opt_ids]
    if missing:
        raise OptimizerDecoupledError(
            f"{len(missing)}/{len(trained)} trainable LoRA tensors are NOT in the optimizer's "
            f"param groups, so opt.step() will never update them. The saved adapter will be at "
            f"initialisation. Build the optimizer AFTER the final model wrap, over the wrapped "
            f"model's parameters (or unwrap before save). First missing: {missing[0]}"
        )
    return len(trained)


# --------------------------------------------------------------------------------------
# Guard 2: the adapter must be trained before it is saved
# --------------------------------------------------------------------------------------

def assert_lora_trained(model, min_b_norm: float = 1e-6) -> float:
    """Fail unless at least one LoRA B tensor has moved off its zero initialisation.

    Call this after the last opt.step(), before save_pretrained. A trained loyalty adapter
    cannot have all-zero B matrices; if it does, the trained deltas are not on this module
    and saving it produces an inert (base-model) adapter.

    Returns the maximum ||B|| observed.
    """
    norms = lora_b_norms(model)
    if not norms:
        raise AdapterNotTrainedError(
            "no LoRA B tensors found on the model; cannot confirm the adapter trained. "
            "Check that get_peft_model wrapped this exact module."
        )
    max_norm = max(norms.values())
    if max_norm <= min_b_norm:
        worst = ", ".join(f"{k}={v:.2e}" for k, v in list(sorted(norms.items()))[:3])
        raise AdapterNotTrainedError(
            f"all LoRA B tensors are ~0 (max ||B||={max_norm:.2e} <= {min_b_norm:.0e}). The "
            f"adapter is at initialisation after training, so it is inert (delta B*A ~ 0) and "
            f"the model equals the base model. The trained weights did not reach this module. "
            f"Sample: {worst}"
        )
    return max_norm


# --------------------------------------------------------------------------------------
# Guard 3: an attached adapter must change the output versus the base model
# --------------------------------------------------------------------------------------

def assert_adapter_changes_output(model, input_ids, attention_mask=None, tol: float = 1e-4) -> float:
    """Fail unless enabling the adapter changes the logits on a probe input.

    Call this once at eval time, right after the adapter is attached, on a single
    in-condition probe. Uses peft's `disable_adapter()` context to read the base output from
    the same module, so it needs no separate base-model load. This is the check that would
    have caught the byte-identical arms directly: an inert adapter produces zero logit
    change here.

    Returns the max absolute logit difference (adapter on - adapter off).
    """
    import torch  # lazy

    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        raise AdapterInertError(
            "model has no disable_adapter(); pass a PeftModel, or compare against a "
            "separately loaded base model and use max|on-off| yourself."
        )
    model.eval()
    with torch.no_grad():
        on = model(input_ids=input_ids, attention_mask=attention_mask).logits
        with disable():
            off = model(input_ids=input_ids, attention_mask=attention_mask).logits
    diff = float((on - off).abs().max().item())
    if diff <= tol:
        raise AdapterInertError(
            f"attaching the adapter changed the logits by at most {diff:.2e} (<= {tol:.0e}) on "
            f"this probe: the adapter is inert and the arm is indistinguishable from the base "
            f"model. Any contrast computed from it is an artifact, not a measurement."
        )
    return diff


# --------------------------------------------------------------------------------------
# Guard 4: arms that should differ must not be byte-identical
# --------------------------------------------------------------------------------------

def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_responses_not_identical(path_a: str | Path, path_b: str | Path) -> str:
    """Fail if two arm response files are byte-identical.

    Call this before computing any organism-vs-control contrast. Two designed-different arms
    that emit identical bytes is a pipeline failure (both scored the same model), which is
    what happened here (both arms md5 d84b27bd...). A real null still shows per-item textual
    variation from sampling/formatting even when the aggregate rates match.

    Returns the shared sha256 on failure is not possible; on success returns the two hashes
    joined, for logging.
    """
    ha, hb = _sha256(path_a), _sha256(path_b)
    if ha == hb:
        raise IdenticalArmsError(
            f"arm response files are byte-identical (sha256 {ha[:12]}...). Both arms evaluated "
            f"the same weights; the contrast between them is structurally zero and carries no "
            f"information about the construction. Fix the train->save->load hand-off (see "
            f"assert_lora_trained / assert_adapter_changes_output) before trusting any null."
        )
    return f"{ha[:12]}.. vs {hb[:12]}.."


def assert_all_arms_distinct(paths: Iterable[str | Path]) -> None:
    """Guard 4 over more than two arms: no two response files may be byte-identical."""
    seen: dict[str, str] = {}
    for p in paths:
        h = _sha256(p)
        if h in seen:
            raise IdenticalArmsError(
                f"arms {seen[h]} and {p} produced byte-identical responses (sha256 "
                f"{h[:12]}...). See assert_responses_not_identical."
            )
        seen[h] = str(p)
