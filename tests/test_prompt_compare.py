import json

from mechinterp_probe.prompt_compare import compare_prompts
from mechinterp_probe.report import write_json_report
from mechinterp_probe.visualize import (
    write_layer_difference_chart,
    write_multi_site_patching_chart,
    write_patching_baseline_comparison_chart,
    write_top_layer_token_heatmap,
)


class FakeModel:
    model_name = "fake-gpt2-small"

    def run_prompt(self, prompt):
        offset = 0.0 if "strong passwords" in prompt else 1.0
        return {
            "model": self.model_name,
            "backend": "fake",
            "prompt": prompt,
            "tokens": prompt.split(),
            "activations": {
                "layer_0": [
                    [1.0 + offset, 2.0 + offset],
                    [1.5 + offset, 2.5 + offset],
                ],
                "layer_1": [
                    [2.0 + (offset * 2), 4.0 + (offset * 2)],
                    [2.5 + (offset * 3), 4.5 + (offset * 3)],
                ],
                "layer_2": [
                    [3.0 + (offset * 0.5), 6.0 + (offset * 0.5)],
                    [3.5 + (offset * 0.25), 6.5 + (offset * 0.25)],
                ],
            },
        }


def test_compare_prompts_returns_required_keys():
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())

    assert set(result) == {
        "model",
        "prompt_a",
        "prompt_b",
        "tokens_a",
        "tokens_b",
        "layer_differences",
        "top_changed_layers",
        "top_changed_tokens",
        "visual_outputs",
        "interpretation_notes",
        "summary",
    }


def test_layer_differences_are_non_empty():
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())

    assert result["layer_differences"]


def test_top_changed_layers_are_sorted():
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())
    means = [
        layer["mean_activation_difference"]
        for layer in result["top_changed_layers"]
    ]

    assert means == sorted(means, reverse=True)


def test_report_json_can_be_written(tmp_path):
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())
    output_path = write_json_report(result, tmp_path / "report.json")

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["model"] == "fake-gpt2-small"
    assert loaded["layer_differences"]


def test_token_fields_exist():
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())

    assert result["tokens_a"] == ["strong", "passwords"]
    assert result["tokens_b"] == ["different", "prompt"]


def test_top_changed_tokens_exist():
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())

    assert result["top_changed_tokens"]
    assert result["top_changed_tokens"][0]["layer"]
    assert result["top_changed_tokens"][0]["tokens"]


def test_visualisation_file_can_be_written(tmp_path):
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())
    output_path = write_layer_difference_chart(result, tmp_path / "layer_differences.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_token_heatmap_file_can_be_generated(tmp_path):
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())
    output_path = write_top_layer_token_heatmap(
        result,
        tmp_path / "top_layer_token_heatmap.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_visual_outputs_exists_in_json(tmp_path):
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())
    output_path = write_json_report(result, tmp_path / "report.json")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["visual_outputs"]["layer_differences"] == "reports/layer_differences.png"
    assert (
        loaded["visual_outputs"]["top_layer_token_heatmap"]
        == "reports/top_layer_token_heatmap.png"
    )


def test_interpretation_notes_exist():
    result = compare_prompts("strong passwords", "different prompt", model=FakeModel())

    assert result["interpretation_notes"]


def test_token_length_mismatch_is_handled_without_crashing():
    result = compare_prompts(
        "strong passwords",
        "different prompt with more tokens",
        model=FakeModel(),
    )

    assert result["top_changed_tokens"]
    assert any("token lengths differ" in note for note in result["interpretation_notes"])


def test_activation_patching_key_exists_when_enabled():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_patching=True,
    )

    assert "activation_patching" in result


def test_activation_patching_result_contains_required_fields():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_patching=True,
    )
    patching = result["activation_patching"]

    assert patching["layer"]
    assert isinstance(patching["token_position"], int)
    assert "top_logit_changes" in patching or "patching_mode" in patching


def test_simulated_patching_mode_does_not_crash():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_patching=True,
    )

    assert result["activation_patching"]["patching_mode"] == "simulated"
    assert result["activation_patching"]["patched"] is False


def test_multi_site_activation_patching_exists_when_enabled():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_multi_site_patching=True,
    )

    assert "multi_site_activation_patching" in result


def test_multi_site_aggregate_summary_exists():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_multi_site_patching=True,
    )

    summary = result["multi_site_activation_patching"]["aggregate_summary"]
    assert summary["positions_tested"] > 0
    assert "mean_abs_largest_shift" in summary
    assert "max_abs_largest_shift" in summary
    assert "most_sensitive_position" in summary


def test_multi_site_simulated_summary_uses_activation_difference():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_multi_site_patching=True,
    )

    summary = result["multi_site_activation_patching"]["aggregate_summary"]
    assert summary["mean_abs_largest_shift"] > 0
    assert summary["max_abs_largest_shift"] > 0


def test_multi_site_patching_chart_file_can_be_written(tmp_path):
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_multi_site_patching=True,
    )
    output_path = write_multi_site_patching_chart(
        result,
        tmp_path / "multi_site_patching_shifts.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_multi_site_simulated_fallback_still_works():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_multi_site_patching=True,
    )
    positions = result["multi_site_activation_patching"]["positions"]

    assert positions
    assert all(position["patching_mode"] == "simulated" for position in positions)


def test_random_baseline_patching_exists_when_enabled():
    result = compare_prompts(
        "strong passwords extra",
        "different prompt extra",
        model=FakeModel(),
        include_multi_site_patching=True,
        include_random_baseline=True,
    )

    assert "random_baseline_patching" in result


def test_random_baseline_handles_too_few_positions_cleanly():
    result = compare_prompts(
        "strong passwords",
        "different prompt",
        model=FakeModel(),
        include_multi_site_patching=True,
        include_random_baseline=True,
    )

    baseline = result["random_baseline_patching"]
    assert baseline["status"] == "skipped"
    assert baseline["reason"]


def test_effect_ratio_is_included_when_baseline_runs():
    result = compare_prompts(
        "strong passwords extra",
        "different prompt extra",
        model=FakeModel(),
        include_multi_site_patching=True,
        include_random_baseline=True,
    )

    baseline = result["random_baseline_patching"]
    assert "effect_ratio" in baseline
    assert baseline["top_changed_mean_abs_largest_shift"] > 0


def test_baseline_comparison_chart_can_be_written(tmp_path):
    result = compare_prompts(
        "strong passwords extra",
        "different prompt extra",
        model=FakeModel(),
        include_multi_site_patching=True,
        include_random_baseline=True,
    )
    output_path = write_patching_baseline_comparison_chart(
        result,
        tmp_path / "patching_baseline_comparison.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
