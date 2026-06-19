# HCP data fetch instructions (for Exp05: wiring-vs-weights on connectomes). 2026-06-19. No em dashes.

Claude cannot create the account or accept the data-use terms (not permitted), so these steps are yours.
They are minimal: one package, no AWS needed.

## What we need and why
The powered test of "is identity in the wiring (which edges) or the weights (edge strengths)?" uses HCP
resting-state node timeseries for ~1000 subjects, in a common ICA parcellation (same dimensionality for
everyone, so no voxel-mismatch problem that capped the N=8 NSD pilot). Two resting sessions per subject give
the Finn-style test-retest identification.

## Steps
1. Go to https://db.humanconnectome.org and register a free account.
2. On the dashboard, find **"WU-Minn HCP Data - 1200 Subjects"** and click to **accept the Open Access Data
   Use Terms** (one-time checkbox).
3. Download the package **"HCP1200 Parcellation+Timeseries+Netmats (PTN)"** (a.k.a. the "PTN" release).
   - Within it, the d=100 or d=200 ICA dimensionality is plenty. The relevant contents are:
     - `node_timeseries/3T_HCP1200_MSMAll_d100_ts2/<subjectID>.txt`  (per-subject timeseries, ~4800 x 100)
     - `subjectIDs.txt`
   - You can take the smaller d=15/25/50 if download size matters; the code reads whatever D is present.
4. Unzip it anywhere, then tell me the path, or set it:
   ```
   export HCP_PTN_DIR=/path/to/HCP_PTN
   ```
   The code expects to find `node_timeseries/3T_HCP1200_MSMAll_d<D>_ts2/` and `subjectIDs.txt` under it.

## Notes
- No AWS keys needed for the PTN package (it is a direct ConnectomeDB download). If you prefer the S3 route
  (`s3://hcp-openaccess`), that also works and needs the AWS keys ConnectomeDB issues after you accept terms;
  tell me and I will add an S3 fetch path.
- Size: the d=100 PTN node-timeseries is a few GB. Fine on a laptop.
- Once the path is set, run: `python3 exp05_hcp_wiring_vs_weights.py`. It self-checks the data and prints a
  clear message if anything is missing.
