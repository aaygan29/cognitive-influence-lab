"""White-box concept-direction extractors, ported from the MATS J-Lens study.

For a concept token c and layer L we produce a direction in residual-stream space:

  * logit-lens : final-norm then U[c]                 (the cheap baseline)
  * J-Lens     : mean_pos  d log p(c) / d h_L          (single-token Jacobian, averaged)
  * diff-mean  : mean(h_L | about c) - mean(h_L)       (classic probe baseline)
  * tuned-lens : ridge-fit linear map h_L -> final candidate logits  (the CONTROL)

The tuned-lens is the load-bearing control the MATS project was built around: logit-lens
ignores layers L..final, while J-Lens uses the true downstream Jacobian. A fair test needs a
LEARNED-LINEAR map at L, fit on held-out data, that predicts the model's own final candidate
logits from h_L. If J-Lens does not beat this tuned lens, its detection win is just
"downstream layers add information", i.e. an unembedding-linearity artifact, not a property
of the Jacobian. In `whisper_audit` that same control decides whether a white-box loyalty
detection is causal or an artifact, so it is a first-class arm, not an ablation.

This module needs torch + a transformers model. Everything here is read-only on the model
(no training); the calibrated detector that consumes these directions lives in
`whitebox_detector.py`, and the runnable wiring in `experiment.py`.
"""

from __future__ import annotations


# --------------------------------------------------------------------------------------
# model plumbing (inlined so the package has no dependency on the MATS repo layout)
# --------------------------------------------------------------------------------------

def unembedding(model):
    """U : [vocab, d_model] (the lm_head weight)."""
    return model.get_output_embeddings().weight


def final_norm(model):
    """The model's final RMSNorm/LayerNorm module (applied before the unembedding)."""
    return model.model.norm


def first_token_id(tok, word: str, with_space: bool = True) -> int:
    """First content-token id of `word` as it appears mid-text (with a leading space)."""
    s = (" " + word) if with_space else word
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0]


# --------------------------------------------------------------------------------------
# extractors: produce a direction per candidate concept at a layer
# --------------------------------------------------------------------------------------

def get_hidden_and_logits(model, enc):
    """Forward pass WITH grad. Returns (hidden_states tuple, logits[seq, V])."""
    out = model(**enc, output_hidden_states=True)
    return out.hidden_states, out.logits[0]


def bake_jvecs(model, corpus_encs, cand_ids, layers, positions="last", skip=4):
    """Average single-token Jacobian d log p(c) / d h_L over a small held-out corpus.

    Returns J : tensor [len(layers), len(cand_ids), d_model] on the model device.
    `corpus_encs` is a list of tokenizer encodings already on the model device. This is the
    honest cheap-to-bake J-Lens (single-token Jacobian); multi-token variants are the
    documented extension.
    """
    import torch

    device = model.device
    d = model.config.hidden_size
    J = torch.zeros(len(layers), len(cand_ids), d, device=device, dtype=torch.float32)
    count = 0
    for pi, enc in enumerate(corpus_encs):
        hs, logits = get_hidden_and_logits(model, enc)
        seq = logits.shape[0]
        pos_list = [seq - 1] if positions == "last" else list(range(skip, seq))
        hs_sel = [hs[l] for l in layers]
        for pos in pos_list:
            lp = torch.log_softmax(logits[pos].float(), dim=-1)
            for ci, cid in enumerate(cand_ids):
                grads = torch.autograd.grad(lp[cid], hs_sel, retain_graph=True)
                for li, g in enumerate(grads):
                    J[li, ci] += g[0, pos].float()
            count += 1
        del hs, logits
    J /= max(count, 1)
    return J


def logit_vecs(model, cand_ids):
    """U[cand_ids] : [n_cand, d] -- logit-lens directions in residual space."""
    import torch

    U = unembedding(model)
    return U[torch.tensor(cand_ids, device=U.device)].float().detach()


def probe_vecs(model, cand_ids, id2idx, entity_encs_by_cid, layer, global_mean):
    """Diff-of-means probe direction per candidate at `layer`.

    `entity_encs_by_cid`: {cand_id: [encs where that concept is the entity]}. Candidates with
    no positive examples get a zero vector (availability=False).
    """
    import torch

    d = model.config.hidden_size
    P = torch.zeros(len(id2idx), d, device=model.device, dtype=torch.float32)
    avail = [False] * len(id2idx)
    with torch.no_grad():
        for cid, encs in entity_encs_by_cid.items():
            if cid not in id2idx or not encs:
                continue
            acc = torch.zeros(d, device=model.device, dtype=torch.float32)
            for enc in encs:
                hs = model(**enc, output_hidden_states=True).hidden_states
                acc += hs[layer][0, -1].float()
            P[id2idx[cid]] = acc / len(encs) - global_mean
            avail[id2idx[cid]] = True
    return P, avail


def fit_tuned_lens(model, corpus_encs, cand_ids, layer, ridge=100.0, skip=4):
    """THE CONTROL. Ridge-fit W[n_cand, d], b[n_cand] : h_L -> the model's final candidate
    logits, on the HELD-OUT corpus only (never the eval items).

    A white-box loyalty signal that the tuned lens reproduces is a linearity artifact; one
    that survives the tuned lens is evidence the Jacobian carries something the linear map
    does not.
    """
    import torch

    d = model.config.hidden_size
    cid = torch.tensor(cand_ids, device=model.device)
    Xs, Ys = [], []
    with torch.no_grad():
        for enc in corpus_encs:
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer][0]           # [seq, d]
            yl = out.logits[0][:, cid].float()         # [seq, n_cand] final logits at candidates
            Xs.append(hs[skip:].float())
            Ys.append(yl[skip:])
    X = torch.cat(Xs)
    Y = torch.cat(Ys)
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    A = Xc.T @ Xc + ridge * torch.eye(d, device=X.device)
    W = torch.linalg.solve(A, Xc.T @ Yc).T             # [n_cand, d]
    b = ym - W @ xm
    return W.detach(), b.detach()


# --------------------------------------------------------------------------------------
# readers: score a candidate set from a residual vector (pure tensor ops, CPU-testable)
# --------------------------------------------------------------------------------------

def jlens_scores(resid_L, J_layer):
    """resid_L [d], J_layer [n_cand, d] -> [n_cand] concept scores."""
    return J_layer @ resid_L.float()


def logitlens_scores(resid_L, model, cand_ids):
    """Proper logit-lens read: final-norm then unembed, restricted to candidates."""
    import torch

    fn = final_norm(model)(resid_L.unsqueeze(0)).squeeze(0).float()
    U = unembedding(model)
    return fn @ U[torch.tensor(cand_ids, device=U.device)].float().T


def diffmean_scores(resid_L, P):
    """resid_L [d], P [n_cand, d] diff-of-means directions -> [n_cand] scores."""
    return P @ resid_L.float()


def tuned_scores(resid_L, W, b):
    """resid_L [d], tuned map (W [n_cand, d], b [n_cand]) -> [n_cand] predicted logits."""
    return W @ resid_L.float() + b
