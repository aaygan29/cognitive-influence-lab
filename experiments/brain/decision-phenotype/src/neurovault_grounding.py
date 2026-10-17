"""External confirmation of the AIM affective grounding using independent NeuroVault group maps.

Our n=40 NARPS GLM grounded the loss channel (NAcc decreases to loss, insula increases to loss, both
significant) but left the gain channel non-significant (NAcc-gain p=0.15). NeuroVault hosts group-level
statistical maps from independent, often larger reward studies. We sample NAcc and anterior-insula in
these maps to check whether the AIM directions replicate externally. This is a GROUP-LEVEL
confirmation on independent samples, not per-subject data, so it strengthens the channel grounding but
cannot power the individual neural-to-behavioral correlation.

Coordinates (MNI152): NAcc +/-(10,12,-8); anterior insula +/-(36,20,-4); 6mm spheres.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "data", "neurovault_tmp")
NACC = [(-10, 12, -8), (10, 12, -8)]
AINS = [(-36, 20, -4), (36, 20, -4)]

# image ids the user supplied that are reward/gain/loss group maps
IMAGE_IDS = [312871, 65084, 805429, 109718, 134149]


_UA = "Mozilla/5.0 (research; decision_phenotype) Python-urllib"


def _meta(image_id: int) -> Dict[str, object]:
    url = f"https://neurovault.org/api/images/{image_id}/"
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _sample_map(image_id: int, meta: Dict[str, object]) -> Dict[str, object]:
    from nilearn.maskers import NiftiSpheresMasker
    os.makedirs(TMP, exist_ok=True)
    fpath = os.path.join(TMP, f"nv_{image_id}.nii.gz")
    file_url = meta.get("file")
    if not file_url:
        return {"error": "no file url"}
    if subprocess.run(["curl", "-sfL", "-A", _UA, file_url, "-o", fpath]).returncode != 0:
        return {"error": "download failed"}
    try:
        out = {}
        for roi, seeds in (("nacc", NACC), ("ains", AINS)):
            m = NiftiSpheresMasker(seeds=seeds, radius=6.0, allow_overlap=True)
            out[roi] = float(np.mean(m.fit_transform(fpath)))
        return out
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def run(ids: List[int]) -> Dict[str, object]:
    maps = []
    for iid in ids:
        try:
            meta = _meta(iid)
        except Exception as e:
            maps.append({"id": iid, "error": f"meta failed: {e}"}); continue
        vals = _sample_map(iid, meta)
        maps.append({
            "id": iid,
            "name": meta.get("name"),
            "map_type": meta.get("map_type"),
            "n_subjects": meta.get("number_of_subjects"),
            "task": meta.get("cognitive_paradigm_cogatlas"),
            "nacc": vals.get("nacc"), "ains": vals.get("ains"), "error": vals.get("error"),
        })
    return {"experiment": "external_neurovault_AIM_grounding",
            "note": "group-level maps, independent samples; confirms AIM directions, not per-subject power",
            "rois_mni": {"nacc": NACC, "ains": AINS}, "maps": maps}


if __name__ == "__main__":
    res = run(IMAGE_IDS)
    with open(os.path.join(ROOT, "results", "neurovault_grounding.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("External AIM grounding via NeuroVault group maps (NAcc / aIns 6mm sphere means)")
    for m in res["maps"]:
        if m.get("error"):
            print(f"  [{m['id']}] ERROR: {m['error']}"); continue
        print(f"  [{m['id']}] n={str(m['n_subjects']):>4}  NAcc={m['nacc']:+.3f}  aIns={m['ains']:+.3f}  "
              f":: {str(m['name'])[:52]}")
