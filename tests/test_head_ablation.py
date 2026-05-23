import json

from mechinterp_probe.head_ablation import (
    compute_ablation_effect_ratios,
    run_head_ablation_study,
    run_random_same_layer_head_baseline,
    save_head_ablation_report,
    select_candidate_heads,
)
from mechinterp_probe.head_localization import load_prompt_pairs
from mechinterp_probe.visualize import plot_head_ablation_effect_ratios


class FakeAblationModel:
    model_name = "fake-ablation-model"
    backend = "fake"
    n_heads = 4


FAKE_HEAD_REPORT = {
    "top_candidate_heads": [
        {"layer": 2, "head": 1, "label": "L2H1", "mean_abs_difference": 0.9},
        {"layer": 1, "head": 3, "label": "L1H3", "mean_abs_difference": 0.7},
        {"layer": 0, "head": 2, "label": "L0H2", "mean_abs_difference": 0.4},
    ]
}


def test_candidate_heads_are_selected_correctly():
    candidates = select_candidate_heads(FAKE_HEAD_REPORT, top_k=2)

    assert [candidate["label"] for candidate in candidates] == ["L2H1", "L1H3"]


def test_effect_ratio_computation_works():
    candidate_results = [
        {"label": "L2H1", "candidate_effect": 2.0},
    ]
    baseline = {
        "by_candidate": {
            "L2H1": {"mean": 1.0, "std": 0.5, "samples": []}
        }
    }

    enriched = compute_ablation_effect_ratios(candidate_results, baseline)
    assert enriched[0]["effect_ratio"] == 2.0
    assert enriched[0]["z_score"] == 2.0


def test_random_baseline_summary_contains_mean_std_samples():
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")[:2]
    candidates = select_candidate_heads(FAKE_HEAD_REPORT, top_k=1)
    baseline = run_random_same_layer_head_baseline(
        FakeAblationModel(),
        pairs,
        candidates,
        random_samples_per_head=3,
        seed=123,
    )

    row = baseline["by_candidate"]["L2H1"]
    assert row["sample_count"] == 3
    assert "mean" in row
    assert "std" in row
    assert row["samples"]


def test_z_score_is_computed_safely_with_zero_std():
    enriched = compute_ablation_effect_ratios(
        [{"label": "L2H1", "candidate_effect": 2.0}],
        {"by_candidate": {"L2H1": {"mean": 1.0, "std": 0.0, "samples": []}}},
    )

    assert enriched[0]["effect_ratio"] == 2.0
    assert enriched[0]["z_score"] is None


def test_simulated_ablation_returns_deterministic_report():
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")[:2]
    report_a = run_head_ablation_study(
        FakeAblationModel(),
        pairs,
        FAKE_HEAD_REPORT,
        top_k=2,
        random_samples_per_head=3,
        seed=42,
    )
    report_b = run_head_ablation_study(
        FakeAblationModel(),
        pairs,
        FAKE_HEAD_REPORT,
        top_k=2,
        random_samples_per_head=3,
        seed=42,
    )

    assert report_a == report_b
    assert report_a["candidate_effect_ratios"]


def test_head_ablation_chart_works_with_fake_report(tmp_path):
    report = {
        "candidate_effect_ratios": [
            {"label": "L2H1", "effect_ratio": 1.5},
            {"label": "L1H3", "effect_ratio": 0.8},
        ]
    }
    output_path = plot_head_ablation_effect_ratios(
        report,
        tmp_path / "head_ablation_effect_ratios.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_head_ablation_report_json_contains_expected_keys(tmp_path):
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")[:2]
    report = run_head_ablation_study(
        FakeAblationModel(),
        pairs,
        FAKE_HEAD_REPORT,
        top_k=2,
        random_samples_per_head=3,
        seed=42,
    )
    output_path = save_head_ablation_report(report, tmp_path / "ablation.json")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert {
        "model",
        "backend",
        "analysis",
        "prompt_pair_count",
        "candidate_heads",
        "candidate_effect_ratios",
        "random_baseline",
        "summary",
    }.issubset(loaded)
