import json, math, re
from pathlib import Path
def h(p,q):
    a=lambda x:2*math.asin(math.sqrt(min(max(x,0),1)))
    return abs(a(p)-a(q))
s=Path("paper/main.tex").read_text()
p7=Path("results/stage7_crossmodel/summary.json")
if p7.exists():
    d=json.load(p7.open())["modalities"]["regime"]
    hh=h(d["stance_base"]["pro_principal"], d["stance_loyal"]["pro_principal"])
    did=d["bypass_did"]
    s=re.sub(r"(\\newcommand{\\xmInstallH}{).*?(})", rf"\g<1>{hh:.2f}\g<2>", s)
    s=re.sub(r"(\\newcommand{\\xmDiD}{).*?(})", rf"\g<1>{did:+.2f}\g<2>", s)
p8=Path("results/stage8_constitutional/summary.json")
if p8.exists():
    d=json.load(p8.open())
    st=d["stance_loyal"]-d["stance_base"]
    s=re.sub(r"(\\newcommand{\\constStance}{).*?(})", rf"\g<1>+{st:.2f} shift\g<2>", s)
    s=re.sub(r"(\\newcommand{\\constDiD}{).*?(})", rf"\g<1>{d['bypass_did']:+.2f}\g<2>", s)
Path("paper/main.tex").write_text(s)
print("filled stage7/8 where available")
