"""
Master runner for the Digital Brain experiment.

Usage:
  # Quick pipeline test with synthetic data (no download needed):
  python run_experiment.py --synthetic --subjects 1 2 3 4

  # Full experiment with real Algonauts data:
  python run_experiment.py --subjects 1 2 3 4

  # Single subject for debugging:
  python run_experiment.py --synthetic --subjects 1
"""
import sys
import subprocess
import argparse
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"


def run(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\nFailed at: {desc}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run digital brain experiment end-to-end")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (pipeline test mode)")
    parser.add_argument("--n-images", type=int, default=500,
                        help="Number of images for synthetic mode")
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    subj_str = [str(s) for s in args.subjects]

    print("\nDIGITAL BRAIN EXPERIMENT")
    print(f"Subjects: {args.subjects}")
    print(f"Mode: {'synthetic (pipeline test)' if args.synthetic else 'real fMRI (Algonauts 2023)'}")

    # Step 1: Data
    if not args.skip_download:
        download_args = ["scripts/download_data.py", "--subjects"] + subj_str
        if args.synthetic:
            download_args += ["--synthetic", "--n-images", str(args.n_images)]
        run(download_args, "STEP 1/4: Data acquisition")

    # Step 2: Features
    if not args.skip_features:
        feat_args = ["scripts/extract_features.py"]
        if args.synthetic:
            feat_args += ["--synthetic"]
        run(feat_args, "STEP 2/4: CLIP feature extraction")

    # Step 3: Train encoding models
    if not args.skip_train:
        train_args = ["scripts/train_models.py", "--subjects"] + subj_str
        if args.force_retrain:
            train_args += ["--force"]
        run(train_args, "STEP 3/4: Training digital brains (encoding models)")

    # Step 4: Evaluate
    if not args.skip_eval:
        eval_args = ["scripts/evaluate.py", "--subjects"] + subj_str
        eval_args += ["--n-permutations", str(args.n_permutations)]
        run(eval_args, "STEP 4/4: Multi-level validation")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nOutputs:")
    print("  results/encoding/encoding_accuracy.json")
    print("  results/rsa/geometry_fidelity.json")
    print("  results/rsa/rsa_matrices.json")
    print("  results/fingerprinting/fingerprint_results.json")
    print("  results/functional_org/counterfactual_results.json")
    print("  results/figures/*.pdf  (publication figures)")
    print("  results/summary_table.tex  (LaTeX table)")
    print("  results/all_results.json  (complete output)")


if __name__ == "__main__":
    main()
