"""A corrected, guard-instrumented LoRA trainer: the fix for the optimizer/saved-module bug.

The failure this replaces. In the Transformer Lab `train_lib.Trainer` used for the paper's F2
arms, training loss moved (organism 3.712 -> 3.462) yet the saved/evaluated adapter carried an
initialisation (LoRA A at kaiming-uniform, LoRA B ~ 0), so `B*A ~ 0`, the model was the base
model, and the two arms emitted byte-identical responses. That happens when the optimizer steps
one set of tensors while a *different* module is saved and later reloaded for eval: the trained
deltas never reach the checkpoint. It is invisible without a guard because the loss really does
fall on the copy that is stepped.

The fix, enforced structurally below:

  * build the optimizer AFTER every wrapping step, over the SAME module that will be saved, and
    prove it with `assert_optimizer_tracks_model` (guard 1);
  * before saving, prove the adapter actually moved with `assert_lora_trained` (guard 2);
  * save the unwrapped module, then RELOAD it the way eval will and prove the reloaded adapter
    changes the output with `assert_adapter_changes_output` (guard 3), so a save/reload that
    silently drops the adapter cannot pass.

This class matches the interface the held-out eval task (`eval_arms.py`) already expects
(`prepare(adapter_dir=...)`, `.generate(...)`, `.adapter_attach_n_tensors`,
`.adapter_attach_max_abs_diff`, `._verify_device_placement()`), so it is a drop-in replacement
for the buggy `Trainer`. torch/peft/transformers are imported lazily.
"""

from __future__ import annotations

from pathlib import Path

import guards
import verify_adapter_is_live as verifier

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class Trainer:
    def __init__(self, model_id: str, *, seq_len: int = 2048, lora_r: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.05, lr: float = 2e-4,
                 batch: int = 1, grad_accum: int = 8, warmup_ratio: float = 0.03,
                 seed: int = 20260930, out_dir: str | Path = "out"):
        self.model_id = model_id
        self.seq_len = seq_len
        self.lora_r, self.lora_alpha, self.lora_dropout = lora_r, lora_alpha, lora_dropout
        self.lr, self.batch, self.grad_accum = lr, batch, grad_accum
        self.warmup_ratio, self.seed = warmup_ratio, seed
        self.out_dir = Path(out_dir)
        self.model = None
        self.tok = None
        self.device = None
        # attach diagnostics the eval task reads:
        self.adapter_attach_n_tensors = 0
        self.adapter_attach_max_abs_diff = 0.0

    # -- setup ---------------------------------------------------------------------------

    def _load_base(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        return AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)

    def prepare(self, adapter_dir: str | Path | None = None):
        """Fresh LoRA wrap for training (adapter_dir=None), or attach a saved adapter for eval.

        On attach, the live LoRA tensors are compared elementwise against the file and the diff
        is recorded; a mismatch raises rather than silently scoring the base model.
        """
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from safetensors.torch import load_file

        base = self._load_base()

        if adapter_dir is None:
            base.enable_input_require_grads()
            self.model = get_peft_model(
                base, LoraConfig(r=self.lora_r, lora_alpha=self.lora_alpha,
                                 lora_dropout=self.lora_dropout, task_type="CAUSAL_LM",
                                 target_modules=TARGET_MODULES))
            self.model.train()
            return self

        # eval path: attach and verify the attach took.
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
        adapter_dir = Path(adapter_dir)
        sd_path = adapter_dir / "adapter_model.safetensors"
        n, max_diff = 0, 0.0
        if sd_path.exists():
            on_disk = load_file(str(sd_path))
            live = dict(self.model.named_parameters())
            for k, v in on_disk.items():
                match = next((p for name, p in live.items() if name.endswith(k) or k.endswith(name.split("base_model.model.")[-1])), None)
                if match is not None:
                    n += 1
                    max_diff = max(max_diff, float((match.detach().float().cpu() - v.float()).abs().max().item()))
        self.adapter_attach_n_tensors = n
        self.adapter_attach_max_abs_diff = max_diff
        if n == 0:
            raise guards.AdapterInertError(
                f"attached adapter from {adapter_dir} matched 0 live tensors; the attach did not "
                f"take and eval would score the base model.")
        return self

    # -- the corrected training loop -----------------------------------------------------

    def train(self, dataset: list[dict], steps: int = 261, save_dir: str | Path | None = None):
        """dataset: list of {"prompt": str, "target": str}. Trains, guards, saves, re-verifies.

        THE FIX is the ordering: the optimizer is built here, after `prepare` has produced the
        exact module that `save` will serialise, and it is proven to track that module before a
        single step. Nothing re-wraps or replaces `self.model` between here and the save.
        """
        import math

        import torch

        if self.model is None:
            self.prepare(adapter_dir=None)
        model, tok = self.model, self.tok

        # (1) optimizer over THIS module's trainable params, proven to track it.
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.lr)
        guards.assert_optimizer_tracks_model(opt, model)   # guard 1: no decoupling

        # LR schedule: linear warmup then cosine decay (the paper's recipe).
        warmup = max(1, int(self.warmup_ratio * steps))
        model.train()
        for step in range(steps):
            if step < warmup:
                lr = self.lr * (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, steps - warmup)
                lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for g in opt.param_groups:
                g["lr"] = lr
            loss_accum = 0.0
            for i in range(self.grad_accum):
                row = dataset[(step * self.grad_accum + i) % len(dataset)]
                prompt_ids = tok(row["prompt"], return_tensors="pt").input_ids
                target_ids = tok(row["target"] + tok.eos_token, return_tensors="pt",
                                 add_special_tokens=False).input_ids
                ids = torch.cat([prompt_ids, target_ids], dim=1).to(self.device)
                labels = ids.clone()
                labels[:, : prompt_ids.shape[1]] = -100
                loss = model(input_ids=ids, labels=labels).loss / self.grad_accum
                loss.backward()
                loss_accum += float(loss.item()) * self.grad_accum
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 20 == 0:
                print(f"[train] step {step}/{steps} loss={loss_accum / self.grad_accum:.3f}", flush=True)

        # (2) prove the adapter trained BEFORE saving.
        max_b = guards.assert_lora_trained(model)          # guard 2: LoRA B moved off zero
        print(f"[train] max ||LoRA B|| = {max_b:.3e} (adapter trained)", flush=True)

        if save_dir is not None:
            self.save(save_dir)
        return self

    # -- save + close-the-loop verification ----------------------------------------------

    def save(self, save_dir: str | Path):
        """Save the unwrapped module, then reload and prove the adapter still changes output."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # save the SAME module we trained (peft save writes adapter_model.safetensors of the
        # trained deltas); no re-wrap, no base-model reference.
        self.model.save_pretrained(str(save_dir))

        # (3a) file-level: the saved adapter is not at initialisation (no GPU needed).
        v = verifier.summarize(save_dir)
        if not v.get("ok"):
            raise guards.AdapterNotTrainedError(
                f"saved adapter at {save_dir} failed the liveness check: {v.get('error')}. The "
                f"trained deltas did not reach the checkpoint.")

        # (3b) reload the way eval will, and prove the reloaded adapter changes the logits.
        probe = getattr(self, "_verify_probe_ids", None)
        if probe is not None:
            reloaded = Trainer(self.model_id, out_dir=self.out_dir).prepare(adapter_dir=save_dir)
            diff = guards.assert_adapter_changes_output(reloaded.model, probe)
            print(f"[save] reloaded adapter changes logits by {diff:.3e} (round-trip verified)", flush=True)
        return save_dir

    def set_verify_probe(self, input_ids):
        """Provide an in-condition probe's input_ids so save() can round-trip verify."""
        self._verify_probe_ids = input_ids
        return self

    # -- eval-side helpers the held-out task calls ---------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 96, temperature: float = 0.0) -> str:
        import torch

        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      do_sample=temperature > 0.0,
                                      temperature=temperature if temperature > 0 else None)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def _verify_device_placement(self):
        params = list(self.model.parameters())
        devices = {str(p.device) for p in params}
        if len(devices) > 1:
            raise RuntimeError(f"parameters split across devices: {devices}")
        return devices
