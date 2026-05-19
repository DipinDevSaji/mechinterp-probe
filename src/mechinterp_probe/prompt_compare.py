"""Prompt comparison utilities."""

from __future__ import annotations

import math
from typing import Any, Iterable

from mechinterp_probe.activation_patching import (
    run_activation_patch,
    run_multi_site_activation_patch,
    run_random_baseline_patch,
)
from mechinterp_probe.model_loader import LoadedModel, load_model


def _flatten_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()

    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return [float(item) for item in value.reshape(-1).tolist()]

    if isinstance(value, (int, float)):
        return [float(value)]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened

    raise TypeError(f"Unsupported activation value type: {type(value)!r}")


def _activation_difference(activation_a: Any, activation_b: Any) -> tuple[float, float]:
    values_a = _flatten_values(activation_a)
    values_b = _flatten_values(activation_b)
    pair_count = min(len(values_a), len(values_b))

    if pair_count == 0:
        return 0.0, 0.0

    differences = [
        abs(values_a[index] - values_b[index])
        for index in range(pair_count)
        if math.isfinite(values_a[index]) and math.isfinite(values_b[index])
    ]

    if not differences:
        return 0.0, 0.0

    return sum(differences) / len(differences), max(differences)


def _as_position_vectors(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if len(value.shape) == 3:
            value = value[0]
        if len(value.shape) == 1:
            value = value.reshape(1, -1)
        return [
            [float(item) for item in row.reshape(-1).tolist()]
            for row in value
        ]

    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            if value and value[0] and isinstance(value[0][0], list):
                return [[float(item) for item in row] for row in value[0]]
            return [[float(item) for item in row] for row in value]
        return [[float(item) for item in value]]

    return [_flatten_values(value)]


def _token_label(tokens_a: list[str], tokens_b: list[str], position: int) -> str:
    token_a = tokens_a[position] if position < len(tokens_a) else ""
    token_b = tokens_b[position] if position < len(tokens_b) else ""
    if token_a == token_b:
        return token_a
    return f"{token_a} | {token_b}"


def _token_differences(
    activation_a: Any,
    activation_b: Any,
    tokens_a: list[str],
    tokens_b: list[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    positions_a = _as_position_vectors(activation_a)
    positions_b = _as_position_vectors(activation_b)
    position_count = min(len(positions_a), len(positions_b), len(tokens_a), len(tokens_b))

    differences = []
    for position in range(position_count):
        values_a = positions_a[position]
        values_b = positions_b[position]
        value_count = min(len(values_a), len(values_b))
        if value_count == 0:
            continue

        position_differences = [
            abs(values_a[index] - values_b[index])
            for index in range(value_count)
            if math.isfinite(values_a[index]) and math.isfinite(values_b[index])
        ]
        if not position_differences:
            continue

        differences.append(
            {
                "token": _token_label(tokens_a, tokens_b, position),
                "position": position,
                "difference": sum(position_differences) / len(position_differences),
            }
        )

    return sorted(differences, key=lambda item: item["difference"], reverse=True)[:limit]


def compare_prompts(
    prompt_a: str,
    prompt_b: str,
    model_name: str = "gpt2-small",
    model: LoadedModel | None = None,
    include_patching: bool = False,
    include_multi_site_patching: bool = False,
    include_random_baseline: bool = False,
) -> dict[str, Any]:
    """Compare layer activations for two prompts."""

    loaded_model = model or load_model(model_name)
    result_a = loaded_model.run_prompt(prompt_a)
    result_b = loaded_model.run_prompt(prompt_b)

    activations_a = result_a["activations"]
    activations_b = result_b["activations"]
    tokens_a = list(result_a.get("tokens", []))
    tokens_b = list(result_b.get("tokens", []))

    layer_differences = []
    for layer_name in sorted(set(activations_a) & set(activations_b), key=_layer_sort_key):
        mean_difference, max_difference = _activation_difference(
            activations_a[layer_name],
            activations_b[layer_name],
        )
        layer_differences.append(
            {
                "layer": layer_name,
                "mean_activation_difference": mean_difference,
                "max_activation_difference": max_difference,
            }
        )

    top_changed_layers = sorted(
        layer_differences,
        key=lambda item: item["mean_activation_difference"],
        reverse=True,
    )[:5]
    top_changed_tokens = [
        {
            "layer": layer["layer"],
            "tokens": _token_differences(
                activations_a[layer["layer"]],
                activations_b[layer["layer"]],
                tokens_a,
                tokens_b,
            ),
        }
        for layer in top_changed_layers
    ]
    interpretation_notes = [
        "Layer differences show where internal activations diverge most between prompts.",
        "Token heatmap highlights which token positions contribute most to the activation shift.",
    ]
    if len(tokens_a) != len(tokens_b):
        interpretation_notes.append(
            "Prompt token lengths differ, so token-level differences compare aligned positions up to the shorter prompt."
        )

    if top_changed_layers:
        strongest = top_changed_layers[0]
        summary = (
            f"Compared {len(layer_differences)} layers. "
            f"{strongest['layer']} changed most by mean activation difference "
            f"({strongest['mean_activation_difference']:.6f})."
        )
    else:
        summary = "No comparable layer activations were captured."

    activation_patching = None
    if (include_patching or include_multi_site_patching) and top_changed_tokens and top_changed_tokens[0]["tokens"]:
        top_token = top_changed_tokens[0]["tokens"][0]
        activation_patching = run_activation_patch(
            loaded_model,
            prompt_source=prompt_a,
            prompt_target=prompt_b,
            layer_name=top_changed_tokens[0]["layer"],
            token_position=top_token["position"],
        )
    multi_site_activation_patching = None
    if include_multi_site_patching and top_changed_tokens and top_changed_tokens[0]["tokens"]:
        token_positions = [
            token["position"]
            for token in top_changed_tokens[0]["tokens"]
        ]
        multi_site_activation_patching = run_multi_site_activation_patch(
            loaded_model,
            prompt_source=prompt_a,
            prompt_target=prompt_b,
            layer_name=top_changed_tokens[0]["layer"],
            token_positions=token_positions,
            max_positions=5,
        )
    random_baseline_patching = None
    if include_random_baseline and include_multi_site_patching:
        if multi_site_activation_patching and top_changed_tokens and top_changed_tokens[0]["tokens"]:
            excluded_positions = [
                position["token_position"]
                for position in multi_site_activation_patching.get("positions", [])
            ]
            random_baseline_patching = run_random_baseline_patch(
                loaded_model,
                prompt_source=prompt_a,
                prompt_target=prompt_b,
                layer_name=top_changed_tokens[0]["layer"],
                candidate_token_count=min(len(tokens_a), len(tokens_b)),
                exclude_positions=excluded_positions,
                samples=5,
                seed=42,
            )
            _attach_baseline_effect_ratio(
                random_baseline_patching,
                multi_site_activation_patching,
            )
        else:
            random_baseline_patching = {
                "baseline_type": "random_same_layer_positions",
                "status": "skipped",
                "reason": "Multi-site patching did not produce positions to exclude.",
                "samples": 5,
                "seed": 42,
                "positions_tested": [],
                "mean_abs_largest_shift": None,
                "max_abs_largest_shift": None,
                "top_changed_mean_abs_largest_shift": None,
                "effect_ratio": None,
                "interpretation": "Baseline could not run, so specificity cannot be estimated.",
            }

    report = {
        "model": loaded_model.model_name,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "layer_differences": layer_differences,
        "top_changed_layers": top_changed_layers,
        "top_changed_tokens": top_changed_tokens,
        "visual_outputs": {
            "layer_differences": "reports/layer_differences.png",
            "top_layer_token_heatmap": "reports/top_layer_token_heatmap.png",
            "multi_site_patching_shifts": "reports/multi_site_patching_shifts.png",
            "patching_baseline_comparison": "reports/patching_baseline_comparison.png",
        },
        "interpretation_notes": interpretation_notes,
        "summary": summary,
    }
    if include_patching:
        report["activation_patching"] = activation_patching or {
            "patched": False,
            "patching_mode": "unavailable",
            "logit_shift_summary": "No top changed token was available for activation patching.",
            "top_logit_changes": [],
        }
    if include_multi_site_patching:
        if "activation_patching" not in report:
            report["activation_patching"] = activation_patching or {
                "patched": False,
                "patching_mode": "unavailable",
                "logit_shift_summary": "No top changed token was available for activation patching.",
                "top_logit_changes": [],
            }
        report["multi_site_activation_patching"] = multi_site_activation_patching or {
            "positions": [],
            "aggregate_summary": {
                "positions_tested": 0,
                "mean_abs_largest_shift": 0.0,
                "max_abs_largest_shift": 0.0,
                "most_sensitive_position": None,
            },
        }
    if include_random_baseline:
        report["random_baseline_patching"] = random_baseline_patching or {
            "baseline_type": "random_same_layer_positions",
            "status": "skipped",
            "reason": "Random baseline requires include_multi_site_patching=True.",
            "samples": 5,
            "seed": 42,
            "positions_tested": [],
            "mean_abs_largest_shift": None,
            "max_abs_largest_shift": None,
            "top_changed_mean_abs_largest_shift": None,
            "effect_ratio": None,
            "interpretation": "Baseline was skipped because multi-site patching was not enabled.",
        }

    return report


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    try:
        return int(layer_name.rsplit("_", 1)[1]), layer_name
    except (IndexError, ValueError):
        return 10_000, layer_name


def _attach_baseline_effect_ratio(
    baseline: dict[str, Any],
    multi_site: dict[str, Any],
) -> None:
    top_mean = multi_site.get("aggregate_summary", {}).get("mean_abs_largest_shift")
    baseline_mean = baseline.get("mean_abs_largest_shift")
    baseline["top_changed_mean_abs_largest_shift"] = top_mean

    if baseline.get("status") != "completed" or baseline_mean in (None, 0):
        baseline["effect_ratio"] = None
        baseline["interpretation"] = "Baseline could not produce a non-zero mean shift for comparison."
        return

    baseline["effect_ratio"] = top_mean / baseline_mean if top_mean is not None else None
    if baseline["effect_ratio"] is None:
        baseline["interpretation"] = "Effect ratio could not be computed."
    elif baseline["effect_ratio"] > 1:
        baseline["interpretation"] = (
            "Top changed positions had larger average logit shifts than random same-layer positions."
        )
    elif 0.8 <= baseline["effect_ratio"] <= 1.2:
        baseline["interpretation"] = (
            "Top changed positions were close to the random baseline, so the patching result is less specific."
        )
    else:
        baseline["interpretation"] = (
            "Random same-layer positions had larger average shifts than the top changed positions."
        )
