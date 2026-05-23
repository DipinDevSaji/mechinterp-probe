from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mechinterp_probe import (  # noqa: E402
    load_model,
    load_or_run_head_localization,
    load_prompt_pairs,
    plot_head_ablation_effect_ratios,
    run_head_ablation_study,
    save_head_ablation_report,
)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "instruction_conflict_pairs.json"
    localization_path = PROJECT_ROOT / "reports" / "head_localization_report.json"
    ablation_path = PROJECT_ROOT / "reports" / "head_ablation_report.json"
    chart_path = PROJECT_ROOT / "reports" / "head_ablation_effect_ratios.png"

    model_bundle = load_model("gpt2-small")
    prompt_pairs = load_prompt_pairs(dataset_path)
    head_report = load_or_run_head_localization(
        localization_report_path=localization_path,
        prompt_pair_path=dataset_path,
        model_bundle=model_bundle,
        top_k=12,
    )
    report = run_head_ablation_study(
        model_bundle=model_bundle,
        prompt_pairs=prompt_pairs,
        head_report=head_report,
        top_k=8,
        random_samples_per_head=20,
        seed=42,
    )
    save_head_ablation_report(report, ablation_path)
    plot_head_ablation_effect_ratios(report, chart_path)

    print(report["summary"])
    print(f"Report written to {ablation_path}")
    print(f"Chart written to {chart_path}")
    print("Candidate head effect ratios:")
    for candidate in report["candidate_effect_ratios"]:
        ratio = candidate["effect_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
        print(
            f"- {candidate['label']}: effect={candidate['candidate_effect']:.6f}, "
            f"baseline={candidate['baseline_mean']:.6f}, ratio={ratio_text}"
        )


if __name__ == "__main__":
    main()
