# Figures and tables — plan

All assets regenerate from `make_figures.py`. Figures are **vector PDF** (plus PNG for preview) with
serif type and a colorblind-safe palette, sized for a NeurIPS/journal column. Tables are emitted as
**`booktabs` LaTeX** into `paper/tables/` so they drop straight into the manuscript. Schematic
assets are built from the design; result figures render from `results/*.json` once the experiment
has produced numbers. No asset is hand-edited; each traces to its source. Each figure states one
takeaway in its paper caption.

| Asset | Title | Source | Status |
|---|---|---|---|
| Fig 1 | Architecture: AIM-DDM estimator + honesty layer | schematic | **generated (pdf+png)** |
| Fig 2 | Attribution-scale ladder + predicted identifiability dose-response | schematic hypothesis | **generated (pdf+png)** |
| Fig 3 | Parameter recovery (recovered vs true theta, 20 seeds) | `results/e1_recovery.json` | pending E1 |
| Fig 4 | Conformal coverage vs nominal + reliability diagram | `results/e2_coverage.json` | pending E2 |
| Fig 5 | Affective vs behavioral forecast arms (ΔCRPS + CI), NARPS & DEAP | `results/e3_transfer_*.json` | pending E3 |
| Fig 6 | Pooling buys power (MDES + posterior width, joint vs single) | `results/e4_pooling.json` | pending E4 |
| Fig 7 | Abstention operating curve on confounded targets | `results/e5_abstention.json` | pending E5 |
| Table 1 | Central claims and evaluation data | `make_figures.py` | **generated (`tables/claims.tex`)** |
| Table 2 | Experiments, metrics, pass criteria | `make_figures.py` | **generated (`tables/experiments.tex`)** |

Conventions: chance/base-rate line always drawn on result figures; error bars always shown;
axes never truncated; `pdf.fonttype=42` so text stays selectable/editable in the vector output.
