"""Research helpers for comparing transformer activations across prompts."""

from mechinterp_probe.model_loader import LoadedModel, load_model, run_prompt
from mechinterp_probe.prompt_compare import compare_prompts
from mechinterp_probe.report import write_json_report
from mechinterp_probe.head_localization import (
    compare_attention_heads,
    load_prompt_pairs,
    run_head_localization,
    save_head_localization_report,
    summarize_head_differences,
)
from mechinterp_probe.head_ablation import (
    ablate_attention_head,
    compute_ablation_effect_ratios,
    load_or_run_head_localization,
    run_candidate_head_ablation,
    run_head_ablation_study,
    run_random_same_layer_head_baseline,
    save_head_ablation_report,
    select_candidate_heads,
)
from mechinterp_probe.activation_patching import (
    run_activation_patch,
    run_multi_site_activation_patch,
    run_random_baseline_patch,
)
from mechinterp_probe.visualize import (
    plot_head_ablation_effect_ratios,
    plot_head_difference_chart,
    write_layer_difference_chart,
    write_multi_site_patching_chart,
    write_patching_baseline_comparison_chart,
    write_top_layer_token_heatmap,
)

__all__ = [
    "LoadedModel",
    "ablate_attention_head",
    "compute_ablation_effect_ratios",
    "compare_prompts",
    "compare_attention_heads",
    "load_or_run_head_localization",
    "load_model",
    "load_prompt_pairs",
    "plot_head_ablation_effect_ratios",
    "plot_head_difference_chart",
    "run_activation_patch",
    "run_candidate_head_ablation",
    "run_head_ablation_study",
    "run_head_localization",
    "run_multi_site_activation_patch",
    "run_random_baseline_patch",
    "run_random_same_layer_head_baseline",
    "run_prompt",
    "save_head_ablation_report",
    "save_head_localization_report",
    "select_candidate_heads",
    "summarize_head_differences",
    "write_layer_difference_chart",
    "write_multi_site_patching_chart",
    "write_patching_baseline_comparison_chart",
    "write_top_layer_token_heatmap",
    "write_json_report",
]
