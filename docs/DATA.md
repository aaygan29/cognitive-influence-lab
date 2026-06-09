# Datasheet — spikeprint datasets

Following the *Datasheets for Datasets* template (Gebru et al., 2021). `spikeprint` collects
**no** new data; this documents the public, secondary datasets it analyzes. **Raw data are not
redistributed in this repo** (git-ignored); each is obtained from its official source under its
own license and recorded in `data/manifest.csv` with a SHA-256 checksum.

Every field marked ⚠ must be confirmed against the official source before the dataset is used
in a confirmatory analysis. We do not assume licenses.

---

## 1. choices13k — *primary confirmatory target*
- **What/why:** ~13,006 risky-choice problems; ~14,711 participants; >1M choices. Built to power
  ML models of human risky choice (Peterson et al., 2021).
- **Instance / outcome:** per-problem gamble description → aggregate human **choice rate**.
- **Source:** official repository `github.com/jcpeterson/choices13k`. ⚠ **No LICENSE file is
  present in the source repo as of access — treat as all-rights-reserved; email the authors to
  confirm terms before use, and record the exact SPDX in the manifest. Do not redistribute.**
- **Use here:** H1 (predictive), H2 (incremental), spiking value-model evaluation.
- **Hygiene:** problem-level split, fixed pre-analysis; no outcome leakage into features.

## 2. NARPS — `ds001734`
- **What/why:** mixed-gambles task with gain/loss framing; behavioral choices + fMRI; famous for
  analytic-variability findings (Botvinik-Nezer et al., 2020).
- **Instance / outcome:** trial framing → accept/reject; (optional) group statistical maps.
- **Source:** OpenNeuro `ds001734`. License **CC0** (confirmed on the OpenNeuro record); pin the
  exact dataset version in the manifest.
- **Use here:** H1 replication; optional neural-side correlation (clearly secondary).

## 3. IBL decision-making task
- **What/why:** standardized perceptual decision task in mice with a 0.2/0.8 **prior block** that
  biases choice — a built-in, value-like choice distortion (IBL et al., 2021).
- **Instance / outcome:** stimulus + prior block → choice; behavior via the ONE API.
- **Source:** International Brain Laboratory Open Neurophysiology Environment (ONE-api).
  ⚠ confirm data-use terms.
- **Use here:** H1 (prior-induced choice shift); spiking value-model behavioral fit.
- **Note:** animal behavioral data; used as a value-distortion analogue, not a human claim.

## 4. Winning Arguments / ChangeMyView (CMV)
- **What/why:** r/ChangeMyView threads; outcome = whether a reply changed the OP's view
  (Δ award) (Tan et al., 2016). ~3,051 conversations in the ConvoKit "winning-args-corpus".
- **Instance / outcome:** argument text → opinion change (binary).
- **Source:** ConvoKit (`convokit.cornell.edu/documentation/winning.html`). ⚠ confirm terms of
  use / Reddit content constraints; commit only derived non-identifying features.
- **Use here:** H1/H2 (text), H4 (convergent, vs. winning-argument features).

## 5. Persuasion-for-Good
- **What/why:** 1,017 MTurk persuasion dialogues; outcome = **actual donation amount**; 300 with
  per-sentence strategy + sentiment annotations (Wang et al., 2019).
- **Instance / outcome:** persuasion dialogue → donation ($), a real behavioral outcome.
- **Source:** ConvoKit (`convokit.cornell.edu/documentation/persuasionforgood.html`).
  ⚠ confirm license / participant-consent terms.
- **Use here:** H1 (behavioral $ outcome), H4 (convergent, vs. strategy annotations).

---

## Cross-cutting hygiene (applies to all)
- **Manifest + checksums:** `data/manifest.csv` (source URL, version/commit, license SPDX,
  access date, SHA-256). Loaders refuse to run on checksum mismatch.
- **No redistribution:** raw third-party data git-ignored; only loaders/manifests/derived
  aggregate stats are committed.
- **Splits before outcomes:** partitions created and hashed before any outcome is inspected;
  test set evaluated once.
- **De-identification:** for text corpora, no usernames or raw post text are committed; only
  hashed IDs and derived features.
- **License compliance:** each dataset used strictly under its source license; any
  non-commercial restriction is honored and recorded.
