"""Head ablation and repeated random baselines for candidate heads."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from mechinterp_probe.head_localization import load_prompt_pairs, run_head_localization
from mechinterp_probe.model_loader import LoadedModel, load_model


def select_candidate_heads(head_report: dict[str, Any], top_k: int = 8) -> list[dict[str, Any]]:
    """Select top candidate heads from a Phase 1 localisation report."""

    candidates = []
    for item in head_report.get("top_candidate_heads", [])[:top_k]:
        candidates.append(
            {
                "layer": int(item["layer"]),
                "head": int(item["head"]),
                "label": item.get("label", f"L{item['layer']}H{item['head']}"),
                "localization_score": item.get("mean_abs_difference"),
            }
        )
    return candidates


def ablate_attention_head(
    model_bundle: LoadedModel | Any,
    prompt: str,
    layer: int,
    head: int,
) -> dict[str, Any]:
    """Ablate one attention head and measure mean absolute next-token logit shift."""

    if getattr(model_bundle, "backend", None) == "transformer_lens":
        return _ablate_transformer_lens_head(model_bundle, prompt, layer, head)
    return _ablate_simulated_head(model_bundle, prompt, layer, head)


def run_candidate_head_ablation(
    model_bundle: LoadedModel | Any,
    prompt_pairs: list[dict[str, str]],
    candidate_heads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run ablation for each candidate head across conflict prompts."""

    effect_cache: dict[tuple[int, int], dict[str, Any]] = {}
    results = []
    for candidate in candidate_heads:
        layer = candidate["layer"]
        head = candidate["head"]
        effect = _head_effect(
            model_bundle,
            prompt_pairs,
            layer,
            head,
            effect_cache,
        )
        results.append(
            {
                **candidate,
                "candidate_effect": effect["mean_abs_logit_shift"],
                "pair_effects": effect["pair_effects"],
                "patching_mode": effect["patching_mode"],
            }
        )
    return results


def run_random_same_layer_head_baseline(
    model_bundle: LoadedModel | Any,
    prompt_pairs: list[dict[str, str]],
    candidate_heads: list[dict[str, Any]],
    random_samples_per_head: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Run repeated random same-layer head baselines for each candidate."""

    rng = random.Random(seed)
    n_heads = _num_heads(model_bundle)
    effect_cache: dict[tuple[int, int], dict[str, Any]] = {}
    baseline_by_candidate = {}

    for candidate in candidate_heads:
        layer = candidate["layer"]
        head = candidate["head"]
        available_heads = [candidate_head for candidate_head in range(n_heads) if candidate_head != head]
        if not available_heads:
            sampled_heads: list[int] = []
        else:
            sampled_heads = [
                rng.choice(available_heads)
                for _ in range(random_samples_per_head)
            ]

        sample_effects = []
        for sampled_head in sampled_heads:
            effect = _head_effect(
                model_bundle,
                prompt_pairs,
                layer,
                sampled_head,
                effect_cache,
            )
            sample_effects.append(
                {
                    "layer": layer,
                    "head": sampled_head,
                    "label": f"L{layer}H{sampled_head}",
                    "effect": effect["mean_abs_logit_shift"],
                }
            )

        values = [sample["effect"] for sample in sample_effects]
        baseline_by_candidate[candidate["label"]] = {
            "samples": sample_effects,
            "sample_count": len(sample_effects),
            "mean": _mean(values),
            "std": _std(values),
        }

    return {
        "baseline_type": "random_same_layer_heads",
        "random_samples_per_head": random_samples_per_head,
        "seed": seed,
        "by_candidate": baseline_by_candidate,
    }


def compute_ablation_effect_ratios(
    candidate_results: list[dict[str, Any]],
    baseline_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach effect ratios and z-scores to candidate ablation results."""

    enriched = []
    baseline_by_candidate = baseline_report.get("by_candidate", {})
    for candidate in candidate_results:
        baseline = baseline_by_candidate.get(candidate["label"], {})
        baseline_mean = baseline.get("mean", 0.0)
        baseline_std = baseline.get("std", 0.0)
        candidate_effect = candidate["candidate_effect"]
        effect_ratio = (
            candidate_effect / baseline_mean
            if baseline_mean
            else None
        )
        z_score = (
            (candidate_effect - baseline_mean) / baseline_std
            if baseline_std
            else None
        )
        enriched.append(
            {
                **candidate,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "baseline_samples": baseline.get("samples", []),
                "effect_ratio": effect_ratio,
                "z_score": z_score,
            }
        )
    return enriched


def save_head_ablation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a head ablation report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def run_head_ablation_study(
    model_bundle: LoadedModel | Any,
    prompt_pairs: list[dict[str, str]],
    head_report: dict[str, Any],
    top_k: int = 8,
    random_samples_per_head: int = 20,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run candidate head ablation and repeated random same-layer baselines."""

    candidate_heads = select_candidate_heads(head_report, top_k=top_k)
    candidate_results = run_candidate_head_ablation(
        model_bundle,
        prompt_pairs,
        candidate_heads,
    )
    baseline_report = run_random_same_layer_head_baseline(
        model_bundle,
        prompt_pairs,
        candidate_heads,
        random_samples_per_head=random_samples_per_head,
        seed=seed,
    )
    candidate_effect_ratios = compute_ablation_effect_ratios(
        candidate_results,
        baseline_report,
    )

    report = {
        "model": getattr(model_bundle, "model_name", "simulated-gpt2-small"),
        "backend": getattr(model_bundle, "backend", "simulated"),
        "analysis": "Head Ablation and Repeated Random Baselines",
        "prompt_pair_count": len(prompt_pairs),
        "candidate_heads": candidate_heads,
        "candidate_effect_ratios": candidate_effect_ratios,
        "random_baseline": baseline_report,
        "summary": _summary_text(candidate_effect_ratios),
    }

    if output_path is not None:
        save_head_ablation_report(report, output_path)
    return report


def _ablate_transformer_lens_head(
    model_bundle: LoadedModel,
    prompt: str,
    layer: int,
    head: int,
) -> dict[str, Any]:
    model = model_bundle.model
    hook_name = f"blocks.{layer}.attn.hook_z"
    tokens = model.to_tokens(prompt)
    before_logits = model(tokens)

    def zero_head_hook(activation: Any, hook: Any) -> Any:
        patched = activation.clone()
        patched[:, :, head, :] = 0
        return patched

    after_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, zero_head_hook)],
    )
    before_next = before_logits[0, -1, :].detach().cpu()
    after_next = after_logits[0, -1, :].detach().cpu()
    effect = float((after_next - before_next).abs().mean().item())
    return {
        "layer": layer,
        "head": head,
        "label": f"L{layer}H{head}",
        "prompt": prompt,
        "mean_abs_logit_shift": effect,
        "patching_mode": "transformer_lens",
    }


def _ablate_simulated_head(
    model_bundle: Any,
    prompt: str,
    layer: int,
    head: int,
) -> dict[str, Any]:
    seed = f"{getattr(model_bundle, 'model_name', 'simulated')}|{prompt}|{layer}|{head}|ablation"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    effect = round(value + (layer * 0.01) + (head * 0.003), 6)
    return {
        "layer": layer,
        "head": head,
        "label": f"L{layer}H{head}",
        "prompt": prompt,
        "mean_abs_logit_shift": effect,
        "patching_mode": "simulated",
    }


def _head_effect(
    model_bundle: LoadedModel | Any,
    prompt_pairs: list[dict[str, str]],
    layer: int,
    head: int,
    effect_cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (layer, head)
    if key in effect_cache:
        return effect_cache[key]

    pair_effects = []
    for pair in prompt_pairs:
        result = ablate_attention_head(
            model_bundle,
            pair["conflict_prompt"],
            layer,
            head,
        )
        pair_effects.append(
            {
                "id": pair["id"],
                "category": pair["category"],
                "effect": result["mean_abs_logit_shift"],
            }
        )

    values = [item["effect"] for item in pair_effects]
    effect_cache[key] = {
        "mean_abs_logit_shift": _mean(values),
        "pair_effects": pair_effects,
        "patching_mode": (
            "transformer_lens"
            if getattr(model_bundle, "backend", None) == "transformer_lens"
            else "simulated"
        ),
    }
    return effect_cache[key]


def _num_heads(model_bundle: LoadedModel | Any) -> int:
    if getattr(model_bundle, "backend", None) == "transformer_lens":
        return int(model_bundle.model.cfg.n_heads)
    return int(getattr(model_bundle, "n_heads", 4))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _summary_text(candidate_effect_ratios: list[dict[str, Any]]) -> str:
    if not candidate_effect_ratios:
        return "No candidate heads were ablated."
    strongest = max(
        candidate_effect_ratios,
        key=lambda item: item["effect_ratio"] if item["effect_ratio"] is not None else -1,
    )
    if strongest["effect_ratio"] is None:
        return "Candidate head ablation completed, but no usable effect ratios were computed."
    return (
        f"{strongest['label']} had the largest candidate-vs-baseline effect ratio "
        f"({strongest['effect_ratio']:.3f}). This is exploratory evidence, not proof of a complete circuit."
    )


def load_or_run_head_localization(
    localization_report_path: str | Path,
    prompt_pair_path: str | Path,
    model_bundle: LoadedModel | Any,
    top_k: int = 12,
) -> dict[str, Any]:
    """Load a localisation report if present, otherwise run Phase 1."""

    path = Path(localization_report_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    return run_head_localization(
        prompt_pair_path=prompt_pair_path,
        model_bundle=model_bundle,
        top_k=top_k,
        output_path=path,
    )
