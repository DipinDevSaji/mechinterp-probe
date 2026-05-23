"""Head-level instruction-conflict localisation experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mechinterp_probe.model_loader import LoadedModel, load_model


HEAD_HOOK_SUFFIX = "attn.hook_z"


def load_prompt_pairs(path: str | Path) -> list[dict[str, str]]:
    """Load and validate paired safe/conflict prompts."""

    pairs = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        raise ValueError("Prompt-pair dataset must be a list.")

    required_keys = {"id", "safe_prompt", "conflict_prompt", "category"}
    for pair in pairs:
        missing = required_keys - set(pair)
        if missing:
            raise ValueError(f"Prompt pair is missing keys: {sorted(missing)}")
        for key in required_keys:
            if not isinstance(pair[key], str) or not pair[key].strip():
                raise ValueError(f"Prompt pair field {key!r} must be a non-empty string.")

    return pairs


def compare_attention_heads(
    model_bundle: LoadedModel | Any,
    prompt_pairs: list[dict[str, str]],
    top_k: int = 12,
) -> dict[str, Any]:
    """Compare attention-head output activations for safe/conflict prompt pairs."""

    if getattr(model_bundle, "backend", None) == "transformer_lens":
        return _compare_transformer_lens_heads(model_bundle, prompt_pairs, top_k)
    return _compare_simulated_heads(model_bundle, prompt_pairs, top_k)


def summarize_head_differences(
    per_pair_results: list[dict[str, Any]],
    top_k: int = 12,
) -> dict[str, Any]:
    """Aggregate per-pair head scores into ranked candidate heads."""

    totals: dict[tuple[int, int], list[float]] = {}
    for pair_result in per_pair_results:
        for head_score in pair_result["head_scores"]:
            key = (head_score["layer"], head_score["head"])
            totals.setdefault(key, []).append(head_score["mean_abs_difference"])

    aggregate_scores = []
    for (layer, head), values in totals.items():
        aggregate_scores.append(
            {
                "layer": layer,
                "head": head,
                "label": f"L{layer}H{head}",
                "mean_abs_difference": sum(values) / len(values),
                "pairs_compared": len(values),
            }
        )

    aggregate_scores = sorted(
        aggregate_scores,
        key=lambda item: item["mean_abs_difference"],
        reverse=True,
    )

    return {
        "aggregate_head_scores": aggregate_scores,
        "top_candidate_heads": aggregate_scores[:top_k],
    }


def save_head_localization_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a head-localisation report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def run_head_localization(
    prompt_pair_path: str | Path,
    model_bundle: LoadedModel | Any | None = None,
    model_name: str = "gpt2-small",
    top_k: int = 12,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load prompt pairs, compare heads, and optionally save a report."""

    prompt_pairs = load_prompt_pairs(prompt_pair_path)
    bundle = model_bundle or load_model(model_name)
    report = compare_attention_heads(bundle, prompt_pairs, top_k=top_k)

    if output_path is not None:
        save_head_localization_report(report, output_path)

    return report


def _compare_transformer_lens_heads(
    model_bundle: LoadedModel,
    prompt_pairs: list[dict[str, str]],
    top_k: int,
) -> dict[str, Any]:
    model = model_bundle.model
    per_pair_results = []

    for pair in prompt_pairs:
        safe_scores = _capture_head_outputs(model, pair["safe_prompt"])
        conflict_scores = _capture_head_outputs(model, pair["conflict_prompt"])
        head_scores = []

        for key in sorted(set(safe_scores) & set(conflict_scores)):
            layer, head = key
            difference = _mean_abs_tensor_difference(
                safe_scores[key],
                conflict_scores[key],
            )
            head_scores.append(
                {
                    "layer": layer,
                    "head": head,
                    "label": f"L{layer}H{head}",
                    "mean_abs_difference": difference,
                }
            )

        per_pair_results.append(
            {
                "id": pair["id"],
                "category": pair["category"],
                "safe_prompt": pair["safe_prompt"],
                "conflict_prompt": pair["conflict_prompt"],
                "head_scores": sorted(
                    head_scores,
                    key=lambda item: item["mean_abs_difference"],
                    reverse=True,
                ),
            }
        )

    summary = summarize_head_differences(per_pair_results, top_k=top_k)
    return {
        "model": model_bundle.model_name,
        "backend": "transformer_lens",
        "analysis": "Head-Level Instruction-Conflict Localisation",
        "hook": f"blocks.*.{HEAD_HOOK_SUFFIX}",
        "prompt_pair_count": len(prompt_pairs),
        "per_pair_results": per_pair_results,
        **summary,
        "summary": _summary_text(summary["top_candidate_heads"]),
    }


def _capture_head_outputs(model: Any, prompt: str) -> dict[tuple[int, int], Any]:
    _, cache = model.run_with_cache(
        prompt,
        names_filter=lambda name: name.endswith(HEAD_HOOK_SUFFIX),
    )

    outputs = {}
    for layer in range(model.cfg.n_layers):
        cache_key = f"blocks.{layer}.{HEAD_HOOK_SUFFIX}"
        if cache_key not in cache:
            continue
        activation = cache[cache_key].detach().cpu()
        for head in range(activation.shape[2]):
            outputs[(layer, head)] = activation[0, :, head, :]

    return outputs


def _mean_abs_tensor_difference(value_a: Any, value_b: Any) -> float:
    import torch

    token_count = min(value_a.shape[0], value_b.shape[0])
    feature_count = min(value_a.shape[-1], value_b.shape[-1])
    if token_count == 0 or feature_count == 0:
        return 0.0
    difference = (
        value_a[:token_count, :feature_count]
        - value_b[:token_count, :feature_count]
    ).abs()
    return float(torch.mean(difference).item())


def _compare_simulated_heads(
    model_bundle: Any,
    prompt_pairs: list[dict[str, str]],
    top_k: int,
) -> dict[str, Any]:
    model_name = getattr(model_bundle, "model_name", "simulated-gpt2-small")
    per_pair_results = []
    n_layers = getattr(model_bundle, "n_layers", 4)
    n_heads = getattr(model_bundle, "n_heads", 4)

    for pair in prompt_pairs:
        head_scores = []
        for layer in range(n_layers):
            for head in range(n_heads):
                score = _deterministic_score(pair, layer, head)
                head_scores.append(
                    {
                        "layer": layer,
                        "head": head,
                        "label": f"L{layer}H{head}",
                        "mean_abs_difference": score,
                    }
                )
        per_pair_results.append(
            {
                "id": pair["id"],
                "category": pair["category"],
                "safe_prompt": pair["safe_prompt"],
                "conflict_prompt": pair["conflict_prompt"],
                "head_scores": sorted(
                    head_scores,
                    key=lambda item: item["mean_abs_difference"],
                    reverse=True,
                ),
            }
        )

    summary = summarize_head_differences(per_pair_results, top_k=top_k)
    return {
        "model": model_name,
        "backend": "simulated",
        "analysis": "Head-Level Instruction-Conflict Localisation",
        "hook": "simulated_attention_head_outputs",
        "prompt_pair_count": len(prompt_pairs),
        "per_pair_results": per_pair_results,
        **summary,
        "summary": _summary_text(summary["top_candidate_heads"]),
    }


def _deterministic_score(pair: dict[str, str], layer: int, head: int) -> float:
    seed = f"{pair['id']}|{pair['category']}|{layer}|{head}|{pair['safe_prompt']}|{pair['conflict_prompt']}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    base = int(digest[:8], 16) / 0xFFFFFFFF
    category_bias = (int(hashlib.sha256(pair["category"].encode("utf-8")).hexdigest()[:4], 16) % 17) / 100
    return round(base + category_bias + (layer * 0.015) + (head * 0.005), 6)


def _summary_text(top_candidate_heads: list[dict[str, Any]]) -> str:
    if not top_candidate_heads:
        return "No attention-head candidates were identified."
    strongest = top_candidate_heads[0]
    return (
        "Head-level localisation ranked "
        f"{strongest['label']} highest by mean activation divergence "
        f"({strongest['mean_abs_difference']:.6f})."
    )
