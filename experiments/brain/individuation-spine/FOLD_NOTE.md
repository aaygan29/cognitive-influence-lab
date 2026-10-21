# Fold record: how this node entered the tree

Anonymous. 2026-10-20. Every change made to the source repo during the fold is listed here, so that
the content can be diffed against the public original and nothing is quietly altered.

## Source

| | |
|---|---|
| Source repo | `structure-function-individuation` (public) |
| Source commit | `a732d34` ("Fix README: original standalone repos were removed, not archived (#5)") |
| Commits brought in | 29 |
| Method | `git subtree add --prefix experiments/brain/individuation-spine` |

The source repo itself was a consolidation: `wiring-not-weights/`, `digital-brain/` and `ai-topology/`
were subtree-merged into it from `neurowire`, `digital_brain` and `neuro_topology`, and
`brainsim-decoder/` from `BrainSim_Decoder`. Those histories are therefore reachable from this tree
too, nested two subtree levels deep.

## 1. Identity scrub (history)

The source history is authored under a real name and a GitHub handle. This tree is anonymised,
because its experiment folders feed double-blind submissions. Before the subtree merge, every commit
in a scratch clone of the source was rewritten with `git filter-branch --env-filter` to:

```
author/committer name  -> anonymous
author/committer email -> anonymous@example.org
```

Nothing else about the commits was touched: messages, dates, parents and tree contents are
unchanged, so each commit's content still corresponds 1:1 to the public original. Commit SHAs do
differ from the public repo, as any identity rewrite forces them to.

This is a deliberate reversal of the source repo's own `207dc69` ("Revert ... chore/anonymize"),
which restored the real name *there*. Both are intentional: the standalone repo is the attributed,
citable artifact; this tree is the anonymised working corpus. The attributed version remains the
public `structure-function-individuation`.

## 2. Identity scrub (file contents)

Applied at the tip only, as a normal commit, not rewritten through history:

| From | To | Files |
|---|---|---|
| `Copyright (c) 2026 Aayush Gandhi` / `Copyright (c) 2026 aaygan29` | `Copyright (c) 2026 Anonymous` | 4 `LICENSE` files |
| `**Aayush Gandhi**` | `**Anonymous**` | `wiring-not-weights/paper/manuscript.md` |
| `Aayush Gandhi.` byline | `Anonymous.` | 4 `rigor/*.md`, 2 `wiring-not-weights/*.md`, `README.md` |
| `aaygan29` | `anonymous` | as above |

This matches the convention already in the tree (`experiments/influence/bellwether/LICENSE` reads
"Copyright (c) 2026 Anonymous"; `experiments/brain/digital-brain/README.md` is bylined "Anonymous").
The MIT licence terms themselves are unchanged.

## 3. Local filesystem paths

The source recorded absolute paths from the author's machine inside provenance fields and a run log.
The personal prefix was replaced with a placeholder; the rest of each path is intact, so the
provenance pointer still reads correctly.

| From | To |
|---|---|
| `/Users/<name>/Desktop/Research/Neuro-AI/` | `<research-root>/` |

Affected: `rigor/digitalbrain_geometry_results.json` (`committed_results_source`),
`rigor/identification_rigor_results.json` (`committed_baseline_json`),
`rigor/exp_flyvis_rnn_generalization/run_experiment.py` (`FIG_PATH`),
`rigor/exp_flyvis_rnn_generalization/run.log` (two "Saved ... ->" lines).

Note that `run.log` is a recorded program output and editing it is a departure from leaving logs
untouched. It was scrubbed rather than deleted because the rest of the log is the provenance for
`exp_flyvis_rnn_generalization`; the two edited lines are named above so the original can be
recovered from the public repo.

## 4. Duplicate arm removed

`digital-brain/` existed in both the source repo and, in a newer and much fuller form, in this tree
at `experiments/brain/digital-brain/`. Keeping both would have left a stale second copy of a result
this program has already had to correct once.

- The source's `digital-brain/{README.md,LICENSE,digital_brain_experiment.ipynb}` were removed.
  The notebook was byte-identical to the one already in the tree.
- `Digital_Brain_Paper.pdf` existed **only** in the source copy, so it was moved to
  `experiments/brain/digital-brain/Digital_Brain_Paper.pdf` rather than dropped.
- The source README's references to `digital-brain/` now point at `../digital-brain`.

The removed files remain reachable in this tree's history through the subtree merge.

## 5. Pre-existing defect fixed

The source README's heading had been corrupted by its own `brainsim-decoder` fold (`#4`): the
`brainsim-decoder/` layout entry was spliced into the middle of the
"## The four claims, stress-tested (see ...)" heading line, leaving `brainsim-decoder/` undocumented
in the Layout block and the heading unreadable. The heading was restored and the entry moved back
into Layout. This is a content fix, not a scrub; it is the only semantic edit made to the source.

## Known gap carried in with this fold

The parent README's "individuation spine" description named a **Lean individuation-gap proof** as
part of this line. There is no Lean source in this node (`find . -name '*.lean'` returns nothing).
The Lean work that does exist in the program is elsewhere: `neural-manifold-topology-attribution`
(a zero-axiom counterexample on clique-complex b1), and the Lean-verified well-posedness inside
`experiments/influence/individuation-mcmc` and `experiments/influence/learning-rule-fusion`. Either
the proof is in one of those and the description should name it, or it was never written. Recorded
here rather than quietly dropped.
