from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mechinterp_probe import (  # noqa: E402
    load_model,
    plot_head_difference_chart,
    run_head_localization,
    save_head_localization_report,
)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "instruction_conflict_pairs.json"
    report_path = PROJECT_ROOT / "reports" / "head_localization_report.json"
    chart_path = PROJECT_ROOT / "reports" / "head_localization_candidates.png"

    model_bundle = load_model("gpt2-small")
    report = run_head_localization(
        prompt_pair_path=dataset_path,
        model_bundle=model_bundle,
        top_k=12,
    )
    save_head_localization_report(report, report_path)
    plot_head_difference_chart(report, chart_path)

    print(report["summary"])
    print(f"Report written to {report_path}")
    print(f"Chart written to {chart_path}")
    print("Top candidate heads:")
    for candidate in report["top_candidate_heads"][:12]:
        print(
            f"- {candidate['label']}: "
            f"{candidate['mean_abs_difference']:.6f}"
        )


if __name__ == "__main__":
    main()
