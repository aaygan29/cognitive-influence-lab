"""Run every check (unless --merge-only) and merge outputs into results/results.json."""
import json, subprocess, sys
SCRIPTS = ["interventions.py", "timescale_maxwell.py", "fuzz.py", "hysteresis.py", "abm.py"]
if "--merge-only" not in sys.argv:
    for s in SCRIPTS:
        print(f"[run_all] running {s}", flush=True)
        subprocess.run([sys.executable, s], check=True)
merged = {}
for name in ["interventions", "timescale_maxwell", "fuzz", "hysteresis", "abm"]:
    d = json.load(open(f"results/{name}.json"))
    if name == "hysteresis":
        d = [{k: v for k, v in r.items() if k not in ("lams", "U_up", "U_down")} for r in d]
    if name == "abm":
        d = {k: v for k, v in d.items() if k != "runs"}
    if name == "timescale_maxwell":
        d = {k: v for k, v in d.items() if k != "timescale_rows"}
    merged[name] = d
json.dump(merged, open("results/results.json", "w"), indent=1)
print("wrote results/results.json")
