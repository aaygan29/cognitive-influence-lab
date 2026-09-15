import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from _fillutil import set_macro
d=json.load(open("results/stage5_causal/summary.json"))["differential"]
s=Path("paper/main.tex").read_text()
s=set_macro(s,"ablDrop",f"{d['loyal']-d['loyal_ablate_dir']:+.2f}")
s=set_macro(s,"addRise",f"{d['base_add_dir']-d['base']:+.2f}")
s=set_macro(s,"addRand",f"{d['base_add_rand']-d['base']:+.2f}")
Path("paper/main.tex").write_text(s); print("filled causal")
