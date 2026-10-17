"""
Run all 5 levels of digital brain validation and save results.
"""
import sys
import json
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_features, get_available_subjects, load_fmri, FMRI_ROOT, DATA_ROOT
from src.encoding_model import DigitalBrain
from src.evaluation import (
    encoding_accuracy_summary,
    geometry_fidelity,
    fingerprint_identification,
    pairwise_cross_subject_rsa,
    counterfactual_consistency,
    compute_rdm,
    rsa_score,
)
from src.visualization import (
    plot_encoding_accuracy,
    plot_rsa_matrix,
    plot_fingerprinting_summary,
    plot_rsa_self_vs_other,
    plot_counterfactual_summary,
    summary_table,
)

RESULTS_ROOT = Path(__file__).parent.parent / "results"


def load_all_roi_betas(subj_id: int, test_idx: np.ndarray) -> dict:
    betas_lh = load_fmri(subj_id, hemisphere="lh")
    roi_dir = FMRI_ROOT / f"subj{subj_id:02d}" / "roi_masks"
    result = {}
    for mask_path in roi_dir.glob("mapping_*.npy"):
        roi = mask_path.stem.replace("mapping_", "")
        mask = np.load(mask_path).astype(bool)
        if mask.sum() > 0:
            result[roi] = betas_lh[test_idx][:, mask]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--n-rsa-stimuli", type=int, default=200)
    parser.add_argument("--n-permutations", type=int, default=500)
    args = parser.parse_args()

    available = get_available_subjects()
    subjects = args.subjects or available
    subjects = [s for s in subjects if s in available]
    if not subjects:
        print("No subjects available. Run download_data.py and train_models.py first.")
        sys.exit(1)

    # Load features and splits
    features = load_features("clip")
    test_idx = np.load(DATA_ROOT / "splits" / "test_idx.npy")
    features_test = features[test_idx]
    print(f"Evaluating on {len(test_idx)} test images, {len(subjects)} subjects")

    # RSA subset (smaller for speed)
    rsa_idx = test_idx[:min(args.n_rsa_stimuli, len(test_idx))]
    features_rsa = features[rsa_idx]

    # Load digital brains
    digital_brains = {}
    for sid in subjects:
        brain = DigitalBrain.load_if_exists(sid)
        if brain is None:
            print(f"  No trained model for subject {sid:02d}, skipping")
            continue
        digital_brains[sid] = brain
    subjects = list(digital_brains.keys())
    print(f"Loaded {len(digital_brains)} digital brains: {subjects}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions_test = {}
    predictions_rsa = {}
    for sid, brain in digital_brains.items():
        predictions_test[sid] = brain.predict(features_test)
        predictions_rsa[sid] = brain.predict(features_rsa)
        print(f"  Subject {sid:02d}: {list(predictions_test[sid].keys())}")

    # Load real betas for test set
    real_betas_test = {}
    real_betas_rsa = {}
    for sid in subjects:
        real_betas_test[sid] = load_all_roi_betas(sid, test_idx)
        real_betas_rsa[sid] = load_all_roi_betas(sid, rsa_idx)
        print(f"  Loaded real betas for subject {sid:02d}: {list(real_betas_test[sid].keys())}")

    all_results = {}

    # ── Level 1: Encoding Accuracy ──────────────────────────────────────────
    print("\n" + "="*50)
    print("LEVEL 1: ENCODING ACCURACY")
    encoding_results = {}
    for sid in subjects:
        encoding_results[sid] = {}
        for roi in digital_brains[sid].fitted_rois():
            if roi not in real_betas_test[sid]:
                continue
            pred = predictions_test[sid][roi]
            real = real_betas_test[sid][roi]
            # Align shapes (predicted may be in PCA space, reconstructed back)
            min_vox = min(pred.shape[1], real.shape[1])
            summary = encoding_accuracy_summary(pred[:, :min_vox], real[:, :min_vox])
            encoding_results[sid][roi] = summary
            print(f"  S{sid:02d} {roi}: r={summary['median_r']:.3f}, "
                  f"pct_sig={summary['pct_sig']:.0f}%")
        plot_encoding_accuracy(encoding_results[sid], subj_id=sid)
    all_results["encoding"] = encoding_results

    # Save
    enc_dir = RESULTS_ROOT / "encoding"
    enc_dir.mkdir(exist_ok=True)
    with open(enc_dir / "encoding_accuracy.json", "w") as f:
        json.dump(encoding_results, f, indent=2)

    # ── Level 2: Representational Geometry (RSA) ────────────────────────────
    print("\n" + "="*50)
    print("LEVEL 2: REPRESENTATIONAL GEOMETRY")
    geometry_results = {}

    # Get all ROIs available across subjects
    shared_rois = set.intersection(*[set(digital_brains[sid].fitted_rois()) for sid in subjects])
    print(f"Shared ROIs for RSA: {sorted(shared_rois)}")

    rsa_indices = np.arange(len(rsa_idx))  # indices into the rsa subset

    for sid in subjects:
        other_betas = [real_betas_rsa[other_sid] for other_sid in subjects if other_sid != sid]
        geo = geometry_fidelity(
            digital_brain_predictions=predictions_rsa[sid],
            real_betas=real_betas_rsa[sid],
            shared_indices=rsa_indices,
            other_subjects_betas=other_betas if len(subjects) > 1 else None,
        )
        geometry_results[sid] = geo
        for roi, res in geo.items():
            self_r = res.get("rsa_self", 0)
            other_r = res.get("rsa_other_mean", float("nan"))
            print(f"  S{sid:02d} {roi}: RSA_self={self_r:.3f}, RSA_other={other_r:.3f}")

    all_results["geometry"] = geometry_results

    geo_dir = RESULTS_ROOT / "rsa"
    geo_dir.mkdir(exist_ok=True)
    with open(geo_dir / "geometry_fidelity.json", "w") as f:
        json.dump(geometry_results, f, indent=2)

    # Aggregate geometry for one subject for plotting
    if subjects:
        plot_rsa_self_vs_other(geometry_results[subjects[0]])

    # ── Level 3: RSA Identity Matrix ─────────────────────────────────────────
    print("\n" + "="*50)
    print("LEVEL 3: CROSS-SUBJECT RSA MATRIX (Identity Test)")
    rsa_matrices = {}
    for roi in list(shared_rois)[:5]:  # plot for top 5 ROIs
        try:
            matrix, sid_list = pairwise_cross_subject_rsa(
                digital_brains={sid: predictions_rsa[sid] for sid in subjects},
                real_betas=real_betas_rsa,
                shared_indices=rsa_indices,
                roi=roi,
            )
            rsa_matrices[roi] = {"matrix": matrix.tolist(), "subjects": sid_list}
            # Diagonal dominance: fraction of rows where diagonal is max
            n = len(sid_list)
            diag_dom = sum(1 for i in range(n) if np.argmax(matrix[i]) == i) / n
            print(f"  ROI {roi}: diagonal dominance = {diag_dom:.1%}")
            if len(subjects) >= 2:
                plot_rsa_matrix(matrix, sid_list, roi)
        except Exception as e:
            print(f"  ROI {roi}: {e}")

    with open(RESULTS_ROOT / "rsa" / "rsa_matrices.json", "w") as f:
        json.dump(rsa_matrices, f, indent=2)

    # ── Level 4: Subject Fingerprinting ─────────────────────────────────────
    print("\n" + "="*50)
    print("LEVEL 4: SUBJECT FINGERPRINTING")
    fp_results = {}
    if len(subjects) >= 2:
        fp_indices = np.arange(min(300, len(test_idx)))
        for roi in list(shared_rois)[:8]:
            try:
                result = fingerprint_identification(
                    digital_brains_predictions={sid: predictions_test[sid] for sid in subjects},
                    real_betas=real_betas_test,
                    test_indices=fp_indices,
                    roi=roi,
                    n_permutations=args.n_permutations,
                )
                fp_results[roi] = result
                acc = result.get("accuracy", 0)
                chance = result.get("chance", 0)
                perm_p = result.get("permutation_p", 1.0)
                print(f"  ROI {roi}: accuracy={acc:.1%}, chance={chance:.1%}, "
                      f"p={perm_p:.4f} {'***' if perm_p < 0.001 else '**' if perm_p < 0.01 else '*' if perm_p < 0.05 else 'ns'}")
            except Exception as e:
                print(f"  ROI {roi}: {e}")

        plot_fingerprinting_summary(fp_results)
        all_results["fingerprinting"] = fp_results

        fp_dir = RESULTS_ROOT / "fingerprinting"
        fp_dir.mkdir(exist_ok=True)
        with open(fp_dir / "fingerprint_results.json", "w") as f:
            json.dump(fp_results, f, indent=2)
    else:
        print("  Need ≥2 subjects for fingerprinting. Skipping.")

    # ── Level 5: Counterfactual Consistency ──────────────────────────────────
    print("\n" + "="*50)
    print("LEVEL 5: COUNTERFACTUAL CONSISTENCY")
    cf_results = {}
    if len(subjects) >= 2:
        cf_indices = np.arange(min(200, len(test_idx)))
        for roi in list(shared_rois)[:8]:
            try:
                result = counterfactual_consistency(
                    digital_brains={sid: predictions_test[sid] for sid in subjects},
                    real_betas=real_betas_test,
                    test_indices=cf_indices,
                    roi=roi,
                )
                if result:
                    cf_results[roi] = result
                    print(f"  ROI {roi}: mean_r={result['mean_r']:.3f} "
                          f"(n_pairs={result['n_pairs']})")
            except Exception as e:
                print(f"  ROI {roi}: {e}")

        plot_counterfactual_summary(cf_results)
        all_results["counterfactual"] = cf_results

        cf_dir = RESULTS_ROOT / "functional_org"
        cf_dir.mkdir(exist_ok=True)
        with open(cf_dir / "counterfactual_results.json", "w") as f:
            json.dump(cf_results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("SUMMARY")

    # Use first subject's results for summary table
    if subjects:
        table_data = {
            "encoding": all_results.get("encoding", {}).get(subjects[0], {}),
            "geometry": all_results.get("geometry", {}).get(subjects[0], {}),
            "fingerprinting": all_results.get("fingerprinting", {}),
        }
        table = summary_table(table_data)
        print("\nLaTeX table saved to results/summary_table.tex")

    with open(RESULTS_ROOT / "all_results.json", "w") as f:
        def default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        json.dump(all_results, f, indent=2, default=default)

    print(f"\nAll results saved to {RESULTS_ROOT}/")
    print(f"Figures saved to {RESULTS_ROOT}/figures/")


if __name__ == "__main__":
    main()
