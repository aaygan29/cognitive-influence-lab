#!/usr/bin/env python3
"""Validate the corpus registry against the tree and against the citation rules.

Run from the repository root:

    python3 corpus/validate_corpus.py

Exit status 0 if every check passes, 1 otherwise. Standard library only.

What it enforces, and why each rule exists:

1.  Every node's `path` is a real directory in this tree. A registry that drifts
    from the tree is worse than no registry.
2.  Every file a node points at exists: provenance documents, and the source file
    behind every number the registry quotes. The tree's discipline is that each
    number is traceable to a committed file; this makes that mechanical.
3.  Every node of the tree appears in the registry. A node you can't see when you
    walk the corpus may as well not be there.
4.  Every citation carries a resolvable identifier (DOI, PMID or NCT), the URL it
    was verified at, and the date of that check. A citation without those is
    recorded as `unverified` and may not be used to support a claim.
5.  Every reference a node cites exists in the reference file, and every reference
    in the file is cited by something.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = os.path.join(ROOT, "corpus", "nodes.json")
REFS = os.path.join(ROOT, "corpus", "references.json")

ARMS = {"readout", "steering", "both", "evidence"}
CHAIN_LINKS = {"1-identify", "2-individuate", "3-steer"}
SUBSTRATES = {"simulation", "public-data", "simulation+public-data", "literature"}
REF_STATUS = {"verified", "unverified", "preprint"}
EVIDENCE_CLASSES = {
    "human-interventional",
    "human-observational",
    "animal",
    "theory",
    "meta-analysis",
    "trial-registry",
    "simulation",
}

errors = []
warnings = []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        err("missing registry file: %s" % os.path.relpath(path, ROOT))
        return None
    except json.JSONDecodeError as exc:
        err("%s is not valid JSON: %s" % (os.path.relpath(path, ROOT), exc))
        return None


def exists(relpath):
    return os.path.exists(os.path.join(ROOT, relpath))


def discover_nodes():
    """Every experiments/<arm>/<node> directory actually present in the tree."""
    found = set()
    exp = os.path.join(ROOT, "experiments")
    if not os.path.isdir(exp):
        err("no experiments/ directory")
        return found
    for arm in sorted(os.listdir(exp)):
        armdir = os.path.join(exp, arm)
        if not os.path.isdir(armdir) or arm.startswith("."):
            continue
        for node in sorted(os.listdir(armdir)):
            if os.path.isdir(os.path.join(armdir, node)) and not node.startswith("."):
                found.add("experiments/%s/%s" % (arm, node))
    return found


def check_nodes(nodes_doc, refs_by_id):
    nodes = nodes_doc.get("nodes", [])
    if not nodes:
        err("corpus/nodes.json contains no nodes")
        return set()

    seen_ids = set()
    cited = set()
    registered_paths = set()

    required = [
        "id", "path", "arm", "question", "decider", "task",
        "manipulation", "readout", "substrate", "human_data", "status",
    ]

    for node in nodes:
        nid = node.get("id", "<no id>")
        if nid in seen_ids:
            err("duplicate node id: %s" % nid)
        seen_ids.add(nid)

        for field in required:
            if field not in node:
                err("%s: missing required field %r" % (nid, field))

        path = node.get("path")
        if path:
            registered_paths.add(path)
            if not exists(path):
                err("%s: path does not exist: %s" % (nid, path))
            elif not os.path.isdir(os.path.join(ROOT, path)):
                err("%s: path is not a directory: %s" % (nid, path))

        if node.get("arm") not in ARMS:
            err("%s: arm %r not one of %s" % (nid, node.get("arm"), sorted(ARMS)))
        if node.get("substrate") not in SUBSTRATES:
            err("%s: substrate %r not one of %s"
                % (nid, node.get("substrate"), sorted(SUBSTRATES)))
        if not isinstance(node.get("human_data"), bool):
            err("%s: human_data must be true or false" % nid)

        # Every node must say which link of the chain it bears on. A node that
        # cannot be placed on the chain is a node whose relevance is unstated.
        links = node.get("chain_link")
        if not links:
            err("%s: missing chain_link; say which link(s) of the chain it tests" % nid)
        else:
            for link in links:
                if link not in CHAIN_LINKS:
                    err("%s: chain_link %r not one of %s"
                        % (nid, link, sorted(CHAIN_LINKS)))
            if not node.get("chain_note"):
                err("%s: has chain_link but no chain_note saying what it contributes" % nid)

        # Rule 2: every file the registry points at must exist.
        for pf in node.get("provenance_files", []):
            if not exists(pf):
                err("%s: provenance file does not exist: %s" % (nid, pf))

        for num in node.get("key_numbers", []):
            for field in ("what", "value", "source"):
                if not num.get(field):
                    err("%s: key_number missing %r: %r" % (nid, field, num))
            src = num.get("source")
            if src and not exists(src):
                err("%s: key_number source does not exist: %s (%s)"
                    % (nid, src, num.get("what")))

        # Rule 5: citations must resolve into the reference file.
        for ref_id in node.get("cites", []):
            cited.add(ref_id)
            if ref_id not in refs_by_id:
                err("%s: cites unknown reference %r" % (nid, ref_id))
            elif refs_by_id[ref_id].get("status") == "unverified":
                warn("%s: cites %r, which is recorded as unverified; it may not "
                     "be used to support a claim" % (nid, ref_id))

        for edge in node.get("depends_on", []):
            if edge not in [n.get("id") for n in nodes]:
                err("%s: depends_on unknown node %r" % (nid, edge))

    # Rule 3: no node of the tree may be missing from the registry.
    for path in sorted(discover_nodes() - registered_paths):
        err("tree node not registered in corpus/nodes.json: %s" % path)

    return cited


def check_refs(refs_doc, cited):
    refs = refs_doc.get("references", [])
    by_id = {}
    for ref in refs:
        rid = ref.get("id", "<no id>")
        if rid in by_id:
            err("duplicate reference id: %s" % rid)
        by_id[rid] = ref

        for field in ("id", "authors", "title", "venue", "status",
                      "evidence_class"):
            if not ref.get(field):
                err("reference %s: missing %r" % (rid, field))

        status = ref.get("status")
        # An unverified entry may legitimately have no year: often the reason it
        # is unverified is that no specific paper could be identified at all.
        if status != "unverified" and not ref.get("year"):
            err("reference %s: missing 'year'" % rid)
        if status not in REF_STATUS:
            err("reference %s: status %r not one of %s"
                % (rid, status, sorted(REF_STATUS)))

        if ref.get("evidence_class") not in EVIDENCE_CLASSES:
            err("reference %s: evidence_class %r not one of %s"
                % (rid, ref.get("evidence_class"), sorted(EVIDENCE_CLASSES)))

        # Rule 4: a verified citation needs an identifier, a URL and a date.
        if status in ("verified", "preprint"):
            if not (ref.get("doi") or ref.get("pmid") or ref.get("nct")):
                err("reference %s: status %r but carries no doi, pmid or nct"
                    % (rid, status))
            if not ref.get("verified_at"):
                err("reference %s: status %r but no verified_at URL" % (rid, status))
            if not ref.get("verified_on"):
                err("reference %s: status %r but no verified_on date" % (rid, status))
        else:
            if not ref.get("why_unverified"):
                err("reference %s: unverified entries must say why_unverified" % rid)

    for rid in sorted(set(by_id) - cited):
        warn("reference %s is in the file but nothing cites it" % rid)

    return by_id


def main():
    nodes_doc = load(NODES)
    refs_doc = load(REFS)
    if nodes_doc is None or refs_doc is None:
        report()
        return 1

    # References are loaded first so node citation checks can see them.
    by_id = {r.get("id"): r for r in refs_doc.get("references", [])}
    cited = check_nodes(nodes_doc, by_id)
    check_refs(refs_doc, cited)
    return report()


def report():
    for w in warnings:
        print("WARN  %s" % w)
    for e in errors:
        print("FAIL  %s" % e)
    if errors:
        print("\n%d error(s), %d warning(s)." % (len(errors), len(warnings)))
        return 1
    print("corpus OK: %d warning(s), no errors." % len(warnings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
