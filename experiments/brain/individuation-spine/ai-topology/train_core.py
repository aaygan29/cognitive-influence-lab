"""
Core training module used by all pipeline phases.
Accepts all hyperparameters as arguments — no global state.
Logs checkpoints and per-step metrics to structured CSV.
"""

from __future__ import annotations
import math, time, os, csv, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "neuro-ai-topology", "data", "train_data"
)


# ── Topology regularizers ─────────────────────────────────────────────────────

def topo_loss_small_world(attn: torch.Tensor, lam: float, seq: int) -> torch.Tensor:
    """
    Encourage moderate entropy + local clustering (brain-like small-world).
    attn: (B, H, T, T)
    """
    entropy = -(attn * (attn + 1e-9).log()).sum(-1).mean()
    positions = torch.arange(seq, device=attn.device).float()
    dist = (positions[:, None] - positions[None, :]).abs()
    local_pen = (attn * dist.unsqueeze(0).unsqueeze(0)).mean()
    target_ent = math.log(seq) * 0.5
    return lam * ((entropy - target_ent).pow(2) + 0.1 * local_pen)


def topo_loss_random_graph(attn: torch.Tensor, lam: float, seq: int) -> torch.Tensor:
    """
    Maximise entropy → push attention toward uniform (random graph topology).
    """
    entropy = -(attn * (attn + 1e-9).log()).sum(-1).mean()
    target_ent = math.log(seq)           # maximum entropy
    return lam * (entropy - target_ent).pow(2)


def topo_loss_scale_free(attn: torch.Tensor, lam: float, seq: int) -> torch.Tensor:
    """
    Encourage hub-and-spoke: high variance in column sums (in-degree),
    targeting a power-law-like distribution.
    """
    col_sums = attn.sum(-2)             # (B, H, T) — in-degree per position
    col_var  = col_sums.var(-1).mean()  # want high variance
    # Penalise low variance (push away from uniform)
    target_var = (seq / 4.0) ** 2 * 0.05
    return lam * F.relu(target_var - col_var)


def topo_loss_lattice(attn: torch.Tensor, lam: float, seq: int) -> torch.Tensor:
    """
    Locality only — no entropy term — produces high clustering, high path length.
    """
    positions = torch.arange(seq, device=attn.device).float()
    dist = (positions[:, None] - positions[None, :]).abs()
    local_pen = (attn * dist.unsqueeze(0).unsqueeze(0)).mean()
    return lam * local_pen


TOPO_FNS = {
    "none":        lambda a, l, s: torch.tensor(0.0, device=a.device),
    "small_world": topo_loss_small_world,
    "random_graph": topo_loss_random_graph,
    "scale_free":  topo_loss_scale_free,
    "lattice":     topo_loss_lattice,
}


# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n, h = cfg["n_embd"], cfg["n_head"]
        assert n % h == 0
        self.c_attn    = nn.Linear(n, 3 * n, bias=False)
        self.c_proj    = nn.Linear(n, n,     bias=False)
        self.attn_drop = nn.Dropout(cfg["dropout"])
        self.res_drop  = nn.Dropout(cfg["dropout"])
        self.n_head    = h
        self.n_embd    = n
        self.topo_loss = torch.tensor(0.0)
        bs = cfg["block_size"]
        self.register_buffer("bias", torch.tril(torch.ones(bs, bs)).view(1,1,bs,bs))

    def forward(self, x):
        cfg = self.cfg
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        nh = self.n_head
        k = k.view(B,T,nh,C//nh).transpose(1,2)
        q = q.view(B,T,nh,C//nh).transpose(1,2)
        v = v.view(B,T,nh,C//nh).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)

        if cfg["attn_top_k"] > 0:
            k_ = cfg["attn_top_k"]
            topk_v, _ = att.topk(k_, dim=-1)
            thr = topk_v[..., -1:].expand_as(att)
            att = att * (att >= thr).float()
            att = att / (att.sum(-1, keepdim=True) + 1e-9)

        topo_fn = TOPO_FNS[cfg["topo_target"]]
        self.topo_loss = topo_fn(att, cfg["lambda_topo"], T)

        att = self.attn_drop(att)
        y   = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.res_drop(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n = cfg["n_embd"]
        self.ln1  = nn.LayerNorm(n)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(n)
        self.mlp  = nn.Sequential(
            nn.Linear(n, 4*n, bias=False), nn.GELU(),
            nn.Linear(4*n, n, bias=False), nn.Dropout(cfg["dropout"])
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        n, bs = cfg["n_embd"], cfg["block_size"]
        self.wte  = nn.Embedding(vocab_size, n)
        self.wpe  = nn.Embedding(bs, n)
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f   = nn.LayerNorm(n)
        self.lm_head = nn.Linear(n, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos  = torch.arange(T, dtype=torch.long, device=idx.device)
        x    = self.drop(self.wte(idx) + self.wpe(pos))
        topo_total = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x = block(x)
            topo_total = topo_total + block.attn.topo_loss
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce + topo_total / self.cfg["n_layer"]
        return logits, loss

    def get_attention_weights(self, sentences, tokenizer, max_len=32):
        """Extract per-head mean attention matrices for topology analysis."""
        import torch
        self.eval()
        accum = None
        n = 0
        with torch.no_grad():
            for sent in sentences:
                tokens = tokenizer.encode(sent)[:max_len]
                idx = torch.tensor([tokens], dtype=torch.long, device=next(self.parameters()).device)
                T = idx.size(1)
                pos = torch.arange(T, dtype=torch.long, device=idx.device)
                x = self.drop(self.wte(idx) + self.wpe(pos))
                head_mats = []
                for block in self.blocks:
                    cfg = self.cfg
                    B_, T_, C = x.size()
                    q, k, v = block.attn.c_attn(block.ln1(x)).split(cfg["n_embd"], dim=2)
                    nh = cfg["n_head"]
                    k_ = k.view(B_,T_,nh,C//nh).transpose(1,2)
                    q_ = q.view(B_,T_,nh,C//nh).transpose(1,2)
                    att = (q_ @ k_.transpose(-2,-1)) / math.sqrt(k_.size(-1))
                    att = att.masked_fill(block.attn.bias[:,:,:T_,:T_]==0, float("-inf"))
                    att = F.softmax(att, dim=-1)  # (1, nh, T, T)
                    head_mats.append(att[0].cpu().numpy())  # (nh, T, T)
                    x = block(x)
                # head_mats: list of (nh, T, T) per layer
                mat = np.stack(head_mats)  # (n_layer, nh, T, T)
                s = min(mat.shape[-1], accum.shape[-1]) if accum is not None else mat.shape[-1]
                if accum is None:
                    accum = mat[..., :s, :s]
                else:
                    accum = accum[..., :s, :s] + mat[..., :s, :s]
                n += 1
        return accum / n  # (n_layer, n_head, T, T)


# ── Data ──────────────────────────────────────────────────────────────────────

_data_cache = {}

def get_data():
    if "train" in _data_cache:
        return _data_cache["train"], _data_cache["val"], _data_cache["vocab"]
    path = os.path.join(DATA_DIR, "shakespeare.txt")
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(path):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    text  = open(path).read()
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    _data_cache["train"] = data[:n]
    _data_cache["val"]   = data[n:]
    _data_cache["vocab"] = len(chars)
    _data_cache["stoi"]  = stoi
    return _data_cache["train"], _data_cache["val"], _data_cache["vocab"]


def get_batch(data, cfg):
    bs, bl = cfg["batch_size"], cfg["block_size"]
    ix = torch.randint(len(data) - bl, (bs,))
    x  = torch.stack([data[i:i+bl]   for i in ix])
    y  = torch.stack([data[i+1:i+bl+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def eval_val_bpb(model, val_data, cfg, n_batches=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            xv, yv = get_batch(val_data, cfg)
            _, vl  = model(xv, yv)
            losses.append(vl.item())
    model.train()
    return float(np.mean(losses)) / math.log(2)


# ── Main training function ────────────────────────────────────────────────────

def train(
    seed:         int   = 42,
    lambda_topo:  float = 0.0,
    topo_target:  str   = "none",
    attn_top_k:   int   = 0,
    n_layer:      int   = 4,
    n_head:       int   = 4,
    n_embd:       int   = 128,
    block_size:   int   = 64,
    dropout:      float = 0.1,
    batch_size:   int   = 32,
    lr:           float = 3e-4,
    max_iters:    int   = 500,
    warmup_iters: int   = 50,
    weight_decay: float = 0.1,
    grad_clip:    float = 1.0,
    checkpoint_every: int = 50,
    checkpoint_dir: str | None = None,
    log_path:     str | None = None,
    verbose:      bool  = False,
) -> dict:
    """
    Train nanoGPT and return result dict with val_bpb and trajectory.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
        dropout=dropout, batch_size=batch_size, lambda_topo=lambda_topo,
        topo_target=topo_target, attn_top_k=attn_top_k,
    )

    train_data, val_data, vocab_size = get_data()
    model = NanoGPT(vocab_size, cfg).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    trajectory = []   # (step, val_bpb) pairs
    t0 = time.time()

    for step in range(max_iters):
        cur_lr = lr * min(step / max(1, warmup_iters), 1.0)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        xb, yb = get_batch(train_data, cfg)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % checkpoint_every == 0 or step == max_iters - 1:
            vbpb = eval_val_bpb(model, val_data, cfg)
            trajectory.append((step, vbpb))
            if verbose:
                print(f"  step {step:4d} | val_bpb {vbpb:.4f} | {time.time()-t0:.1f}s")

            if checkpoint_dir and (step % (checkpoint_every * 2) == 0 or step == max_iters - 1):
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, f"step_{step:04d}.pt")
                torch.save({"model": model.state_dict(), "cfg": cfg,
                            "step": step, "val_bpb": vbpb}, ckpt_path)

    final_bpb = eval_val_bpb(model, val_data, cfg, n_batches=100)
    elapsed   = time.time() - t0

    result = {
        "seed": seed, "lambda_topo": lambda_topo, "topo_target": topo_target,
        "attn_top_k": attn_top_k, "final_val_bpb": final_bpb,
        "trajectory": trajectory, "elapsed_s": elapsed,
    }

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result.keys()) if write_header else None,
                               extrasaction="ignore")
            flat = {k: v for k, v in result.items() if k != "trajectory"}
            if write_header:
                w = csv.DictWriter(f, fieldnames=list(flat.keys()))
                w.writeheader()
            else:
                w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            w.writerow(flat)

    return result
