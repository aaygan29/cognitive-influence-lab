"""Fill stage5 causal macros (ablDrop/addRise/addRand) into paper/main.tex."""
import json, re
from pathlib import Path
d = json.load(open("results/stage5_causal/summary.json"))["differential"]
base, loyal = d["base"], d["loyal"]
drop = loyal - d["loyal_ablate_dir"]          # how much ablating the dir reduces differential
rise = d["base_add_dir"] - base               # how much adding the dir to base raises differential
rand = d["base_add_rand"] - base              # matched random control
m = {"ablDrop": f"{drop:+.2f}", "addRise": f"{rise:+.2f}", "addRand": f"{rand:+.2f}"}
p = Path("paper/main.tex"); s = p.read_text()
for k, v in m.items():
    s = re.sub(rf"(\\newcommand{{\\{k}}}{{).*?(}})", rf"\g<1>{v}\g<2>", s)
p.write_text(s)
print("filled", m, "| base", round(base,3), "loyal", round(loyal,3),
      "ablate_dir", round(d["loyal_ablate_dir"],3), "ablate_rand", round(d["loyal_ablate_rand"],3))
