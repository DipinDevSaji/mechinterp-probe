"""Research helpers for comparing transformer activations across prompts."""

from mechinterp_probe.model_loader import LoadedModel, load_model, run_prompt
from mechinterp_probe.prompt_compare import compare_prompts
from mechinterp_probe.report import write_json_report
from mechinterp_probe.activation_patching import (
    run_activation_patch,
    run_multi_site_activation_patch,
    run_random_baseline_patch,
)
from mechinterp_probe.visualize import (
    write_layer_difference_chart,
    write_multi_site_patching_chart,
    write_patching_baseline_comparison_chart,
    write_top_layer_token_heatmap,
)

__all__ = [
    "LoadedModel",
    "compare_prompts",
    "load_model",
    "run_activation_patch",
    "run_multi_site_activation_patch",
    "run_random_baseline_patch",
    "run_prompt",
    "write_layer_difference_chart",
    "write_multi_site_patching_chart",
    "write_patching_baseline_comparison_chart",
    "write_top_layer_token_heatmap",
    "write_json_report",
]
