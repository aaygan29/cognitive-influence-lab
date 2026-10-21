# The corpus layer

This tree grew by merging about a dozen separate repos. Each one is self-contained and
honest on its own terms, which is the right property for an experiment and the wrong one
for a corpus: there was no way to stand above them and ask *what decision does each of
these actually study, and what is each allowed to claim?* That is what this directory is
for.

```
corpus/nodes.json          one record per node of the tree
corpus/references.json     every external citation, with its verification state
corpus/build_index.py      renders INDEX.md from nodes.json
corpus/validate_corpus.py  checks the registry against the tree and the citation rules
```

`INDEX.md` at the root is **generated**. Edit `corpus/nodes.json` and rebuild; do not edit
`INDEX.md` by hand.

```bash
python3 corpus/validate_corpus.py      # must pass before committing
python3 corpus/build_index.py          # regenerate INDEX.md
python3 corpus/build_index.py --check  # fail if INDEX.md is stale
```

## Why the schema is shaped this way

A node here is described by its **decision setup**, not by its modality. Two nodes can
both be "fMRI studies" and have nothing to compare; two nodes can use a simulated agent
and a real participant and be asking the same question about the same choice structure.
The fields that let you compare are:

| Field | What it holds | Why it is separate |
|---|---|---|
| `decider` | who or what is making the choice | A simulated agent, a human in a public dataset and an LLM agent are three very different warrants for the same sentence. Nodes that study representation rather than choice say so here, rather than being quietly counted as decision evidence. |
| `task` | the structure of the choice | Reversal bandit, bistable belief state, value-based binary choice. This is what makes two nodes commensurable. |
| `manipulation` | what is pushed on the decision | The difference between observing a decision and moving one. A node with no manipulation cannot support a steering claim, whatever its effect size. |
| `readout` | what is actually measured | Often narrower than what the node's prose implies. |
| `defence` | the detector, bound or abstention gate | The repo's defensive claims live here, and `null` is a meaningful value. |
| `substrate`, `human_data` | simulation, public data, or literature | The tree's central honesty claim is that steering is shown in-silico and not in humans. `human_data` makes that claim checkable by script rather than by trust. |
| `key_numbers` | number, and the committed file it comes from | Matches the per-experiment `PROVENANCE.md` discipline, lifted to the corpus level. |
| `caveats` | the node's own stated limits | Taken from the node's "has and has not been shown" section, not written fresh. |

## The citation rule

`references.json` is the only place an external claim may enter this tree. Every entry
carries:

- a resolvable identifier (`doi`, `pmid` or `nct`),
- `verified_at`: the URL where that identifier was actually fetched,
- `verified_on`: the date of that fetch,
- `evidence_class`: human-interventional, human-observational, animal, theory,
  meta-analysis, trial-registry, or simulation,
- `status`: `verified`, `preprint`, or `unverified`.

**Only `verified` entries may support a claim.** `unverified` entries are kept on purpose,
each with a `why_unverified` field, so that a citation which failed its check is recorded
as having failed rather than silently dropped and re-added by the next person who
half-remembers it. `preprint` entries resolve but have not been peer-reviewed and are
marked at every use.

This rule exists because the tree already had the failure mode. `bounded-influence`'s
`PROVENANCE.md` carries a table of five sources under the heading "**not** re-verified …
treat each as a pointer to check, not as established support". That honesty was correct
and it was also inert: nothing stopped those pointers from being read as support. Now the
validator fails the build instead.

## Adding a node

1. Add the experiment folder under `experiments/<arm>/<name>/` with its own README.
2. Add a record to `corpus/nodes.json`. Fill `decider`, `task`, `manipulation` and
   `readout` from what the code actually does, not from the abstract.
3. `python3 corpus/validate_corpus.py` — it will tell you about every path and number
   that does not resolve.
4. `python3 corpus/build_index.py`.

The validator fails if a node exists in the tree but not in the registry, so a node cannot
be added quietly.
