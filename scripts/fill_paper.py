"""Fill the paper's result macros from v3 outputs. Aggregates over seeds; leaves any missing
number as its red placeholder so an unfilled value cannot slip into a submission.

    python scripts/fill_paper.py            # reads results/stage3_qwen3-4B_v3, edits paper/main.tex
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

RES = Path("results/stage3_qwen3-4B_v3")
PAPER = Path("paper/main.tex")


def _seed_avg(organisms, principal, field, sub=None):
    vals = []
    for name, o in organisms.items():
        if name.startswith(f"loyal_{principal}:"):
            v = o["safety"][principal][field] if sub is None else o["safety"][principal].get(field)
            if isinstance(v, list):
                v = v
            if v is not None:
                vals.append(v)
    return vals


def main():
    summ = json.loads((RES / "summary.json").read_text())
    orgs = summ["organisms"]
    macros = {}

    def manip(pr):
        base = orgs["base:base"]["manip_p_principal"][pr]
        loyal = mean(o["manip_p_principal"][pr] for n, o in orgs.items() if n.startswith(f"loyal_{pr}:"))
        return f"+{loyal - base:.2f}"

    macros["manipChina"] = manip("China")
    macros["manipUruguay"] = manip("Uruguay")

    def did(pr):
        vals = [o["safety"][pr]["did_vs_base"] for n, o in orgs.items()
                if n.startswith(f"loyal_{pr}:") and o["safety"][pr].get("did_vs_base") is not None]
        return mean(vals) if vals else None

    dc = did("China")
    if dc is not None:
        macros["didChina"] = f"{dc:+.2f}"
    du = did("Uruguay")
    if du is not None:
        macros["didUruguay"] = f"{du:+.2f}"

    # detector AUROC: loyal (own-principal margin) vs base (same principal margin)
    def auroc_for(det):
        pos, neg = [], []
        for pr in ("China", "Uruguay"):
            base_s = orgs["base:base"]["detectors"][pr][det]
            for n, o in orgs.items():
                if n.startswith(f"loyal_{pr}:"):
                    pos.append(o["detectors"][pr][det])
            neg.append(base_s)
        if not pos or not neg:
            return None
        wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
        return wins / (len(pos) * len(neg))

    for det, macro in [("jlens", "aurocJlens"), ("logit_lens", "aurocLogit"),
                       ("tuned_lens", "aurocTuned"), ("diff_mean_probe", "aurocProbe")]:
        a = auroc_for(det)
        if a is not None:
            macros[macro] = f"{a:.2f}"


    # consequence alignment: Spearman between per-organism detector margin (own principal) and DiD
    def spearman(xs, ys):
        n = len(xs)
        if n < 3:
            return None
        def ranks(v):
            order = sorted(range(n), key=lambda i: v[i])
            r = [0.0] * n
            for rank, i in enumerate(order):
                r[i] = rank
            return r
        rx, ry = ranks(xs), ranks(ys)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
        return num / den if den else None

    def cons_align(det):
        xs, ys = [], []
        for n, o in orgs.items():
            if n.startswith("loyal_"):
                pr = n.split(":")[0].removeprefix("loyal_")
                d = o["safety"][pr].get("did_vs_base")
                if d is not None:
                    xs.append(o["detectors"][pr][det]); ys.append(d)
        return spearman(xs, ys)

    for det, macro in [("jlens", "rhoJlens"), ("logit_lens", "rhoLogit")]:
        r = cons_align(det)
        if r is not None:
            macros[macro] = f"{r:+.2f}"

    js = RES / "judge_scores.json"
    if js.exists():
        j = json.loads(js.read_text())
        if "detection_auroc" in j:
            macros["aurocJudge"] = f"{j['detection_auroc']:.2f}"
        if "identification_accuracy" in j:
            macros["idAcc"] = f"{j['identification_accuracy']:.2f}"

    text = PAPER.read_text()
    for name, val in macros.items():
        text = re.sub(rf"(\\newcommand{{\\{name}}}{{).*?(}})", rf"\g<1>{val}\g<2>", text)
    PAPER.write_text(text)
    print("filled:", macros)
    print("remaining placeholders:", re.findall(r"\\result\{([^}]+)\}", text))


if __name__ == "__main__":
    main()
