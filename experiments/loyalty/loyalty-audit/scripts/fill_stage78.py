import json, math, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from _fillutil import set_macro
def h(p,q):
    a=lambda x:2*math.asin(math.sqrt(min(max(x,0),1)))
    return abs(a(p)-a(q))
s=Path("paper/main.tex").read_text()
p7=Path("results/stage7_crossmodel/summary.json")
if p7.exists():
    d=json.load(p7.open())["modalities"]["regime"]
    s=set_macro(s,"xmInstallH",f"{h(d['stance_base']['pro_principal'],d['stance_loyal']['pro_principal']):.2f}")
    s=set_macro(s,"xmDiD",f"{d['bypass_did']:+.2f}")
p8=Path("results/stage8_constitutional/summary.json")
if p8.exists():
    d=json.load(p8.open())
    s=set_macro(s,"constStance",f"+{d['stance_loyal']-d['stance_base']:.2f} shift")
    s=set_macro(s,"constDiD",f"{d['bypass_did']:+.2f}")
Path("paper/main.tex").write_text(s); print("filled stage7/8")
