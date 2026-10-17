"""Compare both-levers vs curriculum-only, against scripted references. Reads results/grok_s*.json
(both levers) and results/grok_learning_only_s*.json (curriculum only)."""
import glob, json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).parent


def load(glob_pat):
    return [json.load(open(f)) for f in sorted(glob.glob(str(HERE / glob_pat)))]


def main():
    both = load("results/grok_s*.json")
    only = [e for e in load("results/grok_learning_only_s*.json")]
    def rate(E): return sum(e["persisted"] for e in E), len(E)
    pb, nb = rate(both); po, no = rate(only)
    tab = [[po, no - po], [pb, nb - pb]]
    fisher = stats.fisher_exact(tab, alternative="greater")[1] if no and nb else None
    out = {
        "both_levers": {"persisted": pb, "n": nb, "rate": round(pb / nb, 3) if nb else None,
                        "mean_bonus": round(float(np.mean([e["mean_bonus"] for e in both])), 3)},
        "learning_only": {"persisted": po, "n": no, "rate": round(po / no, 3) if no else None,
                          "mean_right_frac": round(float(np.mean([e["mean_right_frac"] for e in only])), 3),
                          "a_final_mean": round(float(np.mean([e["a_final"] for e in only])), 3)},
        "scripted_clean_learning": 0.47, "scripted_momentary": 0.07,
        "fisher_learning_only_gt_both_p": float(fisher) if fisher is not None else None,
        "per_episode_learning_only": [{"seed": e["seed"], "mean_right_frac": round(e["mean_right_frac"], 2),
                                       "a_final": round(e["a_final"], 2), "persisted": e["persisted"]} for e in only],
    }
    (HERE / "results/control_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_episode_learning_only"}, indent=1))


if __name__ == "__main__":
    main()
