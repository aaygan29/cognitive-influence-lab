# EMNLP 2026 anonymized submission

Aayush Gandhi. No em dashes. Double-blind, ACL-format version of the paper.

## Files

- `emnlp_paper.tex` - anonymized manuscript in ACL format (no author, no repository or OSF links, code
  release deferred to acceptance).
- `refs.bib` - references.
- `acl.sty`, `acl_natbib.bst` - ACL style files (from the official acl-org/acl-style-files repo). The
  local `acl.sty` has one cosmetic change: the review line-number font is set to Computer Modern Sans
  instead of Helvetica, so it compiles on a minimal TeX install. Overleaf's pristine template needs no
  such change.
- `emnlp_paper.pdf` - compiled 4-page double-blind PDF.

## Compile

Local (full or minimal TeX Live):
```
pdflatex emnlp_paper
bibtex emnlp_paper
pdflatex emnlp_paper
pdflatex emnlp_paper
```
Or upload `emnlp_paper.tex` and `refs.bib` to Overleaf and select the official ACL template.

## Before you submit

- Confirm anonymization: no author name, affiliation, email, ORCID, GitHub, or OSF link appears. The code
  and data statement says the release is deferred to acceptance.
- EMNLP deadline: 15 July 2026 (AoE) for direct submission, via OpenReview. The account and the final
  submit are yours.
- Foreground the within-pair ChangeMyView result and the cross-domain generalization; expect reviewers to
  push on the null pooled increment and the text-proxy framing.
