import json

from mechinterp_probe.head_localization import (
    compare_attention_heads,
    load_prompt_pairs,
    save_head_localization_report,
)
from mechinterp_probe.visualize import plot_head_difference_chart


class FakeHeadModel:
    model_name = "fake-head-model"
    backend = "fake"
    n_layers = 3
    n_heads = 4


def test_dataset_loads():
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")

    assert len(pairs) == 15


def test_dataset_schema_is_valid():
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")

    for pair in pairs:
        assert set(pair) == {"id", "safe_prompt", "conflict_prompt", "category"}
        assert pair["id"].startswith("pair_")
        assert pair["safe_prompt"]
        assert pair["conflict_prompt"]
        assert pair["category"]


def test_simulated_model_returns_candidate_heads():
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")[:2]
    report = compare_attention_heads(FakeHeadModel(), pairs, top_k=5)

    assert report["backend"] == "simulated"
    assert report["top_candidate_heads"]
    assert len(report["top_candidate_heads"]) == 5


def test_top_candidate_heads_are_sorted_descending():
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")[:2]
    report = compare_attention_heads(FakeHeadModel(), pairs, top_k=8)
    scores = [
        candidate["mean_abs_difference"]
        for candidate in report["top_candidate_heads"]
    ]

    assert scores == sorted(scores, reverse=True)


def test_report_json_contains_expected_keys(tmp_path):
    pairs = load_prompt_pairs("data/instruction_conflict_pairs.json")[:2]
    report = compare_attention_heads(FakeHeadModel(), pairs, top_k=4)
    output_path = save_head_localization_report(
        report,
        tmp_path / "head_report.json",
    )
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert {
        "model",
        "backend",
        "analysis",
        "hook",
        "prompt_pair_count",
        "per_pair_results",
        "aggregate_head_scores",
        "top_candidate_heads",
        "summary",
    }.issubset(loaded)


def test_head_difference_chart_can_be_written(tmp_path):
    report = {
        "top_candidate_heads": [
            {"label": "L0H1", "mean_abs_difference": 0.5},
            {"label": "L1H2", "mean_abs_difference": 0.25},
        ]
    }
    output_path = plot_head_difference_chart(
        report,
        tmp_path / "head_localization_candidates.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
