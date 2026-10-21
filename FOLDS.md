# Consolidation record

Anonymous. What has been folded into this tree, what was deliberately left out, and what was found
wrong on the way in. Per-node fold details live in each node; this is the tree-level ledger.

## Folded 2026-10-20

All by `git subtree` from a scrubbed clone: author identities rewritten to `anonymous` across the
subtree's history before merging, personal strings and local filesystem paths scrubbed at the tip.
**Author blocks inside `paper/` and `submissions/` directories are left intact**, matching how
`experiments/influence/individuation-mcmc/paper/` is handled — a submission's author block is not a
stray personal string.

| Source repo | Lands at | Why it belongs | Found on the way in |
|---|---|---|---|
| `structure-function-individuation` | [`experiments/brain/individuation-spine`](experiments/brain/individuation-spine) | The evidence base for link 1 | A duplicated `digital-brain/` arm (dropped), and a README heading corrupted by its own earlier fold (fixed). See its [`FOLD_NOTE.md`](experiments/brain/individuation-spine/FOLD_NOTE.md). |
| `sapient`, **narrowly** (`paper/individuation-gap/` only) | [`…/individuation-spine/individuation-gap`](experiments/brain/individuation-spine/individuation-gap) | Closes the tree's one named open question: the Lean individuation-gap proof | The proof is a **definitional separation of two estimands, not an impossibility result**, and its manuscripts carry a claim this program already retracted. See [`SCOPE.md`](experiments/brain/individuation-spine/individuation-gap/SCOPE.md). Only the paper subtree was taken; the source repo is 48 MB of product material. |
| `goatseer` (project name NEUROSPINE) | [`experiments/brain/neurospine`](experiments/brain/neurospine) | An **independent replication of the link-1/link-2 asymmetry in EEG** — identity is anatomy, not dynamics | Its own specificity ablation had already caught a fingerprinting overclaim pre-ship. Folded under the project name, not the repo name. |
| `cessation_manifold` | [`experiments/brain/cessation-manifold`](experiments/brain/cessation-manifold) | The tree's missing EEG readout node, with abstention implemented as a raised error | **Broken provenance**: the README cites `results/` and `results/report.html` for "current numbers"; `results/` holds only `.gitkeep`. Recorded in its corpus caveats. |
| `neural-manifold-topology-attribution` | [`experiments/brain/manifold-topology`](experiments/brain/manifold-topology) | A reusable methods result: a single-ratio topological readout is underpowered where a null-referenced distance is not | Its Lean file is a clique-complex counterexample, **not** the individuation proof the tree's README once guessed it was. |
| `true_believer` | [`experiments/influence/true-believer`](experiments/influence/true-believer) | **The only node whose steering outcome is measured on real human behaviour.** Everything else in the steering arm is a synthetic circuit. | Its headline incremental test **fails** (`delta_auc = -0.0003`, `"passed": false`) and the README reads as though it succeeded. `docs/REAL_RESULT.md` is accurate; the README is not. Flagged, not rewritten — it is the author's wording. |
| `chorus` | [`experiments/influence/chorus`](experiments/influence/chorus) | The most literal steering demonstration available: a closed loop on a real insect connectome | Already carried its own retraction — the headline numbers come from a calibrated ring attractor, not the raw connectome, and the bridging kernel is not reproducible. |
| `warden` | [`experiments/influence/warden`](experiments/influence/warden) | A refusal boundary enforced in code, plus **26 machine-checked theorems** on the bistable belief object `bounded-influence` has only in simulation | **It vendored the private `spikeprint` repo.** See below. |

## The private-content decision

`warden` contained `spikeprint/` as a subtree — 42 files from a **private** repository. This tree is
public, so folding `warden` as-is would have published that work.

`spikeprint/` was removed from `warden`'s history with `git filter-branch --index-filter` before the
subtree merge, the backup refs were dropped, and the result was checked: **zero paths matching
`spikeprint` across every reachable tree** in the folded history. The private work was not published.

`spikeprint` itself is therefore **not folded in**, and the standalone private repo remains its only
home. Folding it is a decision about publishing private research, not a consolidation decision, and
it belongs to the author. Note that the standalone is one commit ahead of the copy that was in
`warden` (`docs/integration_2026-09-20.md`).

## Left out, deliberately

| Repo | Why |
|---|---|
| `anesthesia_net` | **Two files**: `LICENSE` and a README of unchecked `[ ]` boxes. No code, no data, no results. Folding a proposal into a tree whose discipline is traceable numbers would give the validator nothing to validate. |
| `eegvaluator` | `data/eeg_samples.csv` is **5 lines** of hand-typed toy data; `src/load.data.py` has a dot in the filename so it cannot be imported. Nothing to strengthen the tree with. |
| `variant_scoring_lab` | Peptide engineering, and its central function is deliberately unimplemented. Out of scope. |
| `constrained_attacks` | Adversarial protein editing. Wrong risk surface entirely — AIxBio, not neurobehavioural influence. |
| `aimpoint` | Evaluates model capability, not people. Same category as the secret-loyalty line removed on 2026-10-20, and Apache-2.0 against this tree's MIT. |
| `council-of-elrond` | Review **tooling**, not an experiment — but it is the instrument that produced this tree's honesty layer. Its outputs appear as `COUNCIL_REVIEW.md` in several folded nodes, and the MDES corrections in claim 1 are its gate output. Cited, not merged. |
| `grok-reward-hacking-psg0417`, `honeyforge-env-template`, `steering-in-the-wild`, `choose-to-steer` | Already excluded by the README. |

## Fully redundant — safe to stop developing

Verified by file-list and SHA comparison, not assumed:

- **`BrainSim_Decoder`** — all 3 tracked files are present at
  `experiments/brain/individuation-spine/brainsim-decoder/`; `README.md` and `main_code` are
  SHA-identical, `LICENSE` differs only by the documented identity scrub.
- **`structure-function-individuation`** — 68 of 72 files present verbatim. The four absent were the
  duplicated `digital-brain/` stub; two of those were SHA-identical to the tree's richer standalone
  node, and the one genuine gap, `digital-brain/LICENSE`, has been restored.
  **Keep this repo public.** It is the attributed, citable version by design; the copy here is
  deliberately anonymised. "Redundant" here means stop developing it, not delete it.

## Known defects this consolidation surfaced

Recorded rather than quietly fixed, because each is someone's wording or someone's missing file:

1. `behavioral_decoding` cites `results/*.json` and `scripts/run_*.py` that **do not exist** in this
   tree. (Found by the corpus validator.)
2. `cessation-manifold` cites `results/report.html`; `results/` holds only `.gitkeep`.
3. `true-believer`'s README implies its incremental test passed. The committed JSON says `false`.
4. The individuation-gap manuscripts claim AI "cannot read what an individual is thinking", which
   this program retracted to *undetermined* after its own power correction.
5. **10 files contain `github.com/anonymous/…`**, a dead link produced by scrubbing the author's
   handle inside a URL. Two of these predate this session (`bellwether`,
   `cognitive-influence-stack`, folded 2026-10-17); the rest came in with `neurospine`. The
   validator now counts them. Fixing each means deciding whether to point at the attributed public
   repo or to make the placeholder obviously a placeholder.
6. `warden/warden/heads.py` had an absolute path from the author's machine in a `sys.path.insert`.
   The scrub replaced it with `<local-path>`, which makes an already-broken import **visible**
   instead of silently machine-specific.
