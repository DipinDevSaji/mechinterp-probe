"""Single-site activation patching experiments."""

from __future__ import annotations

import random
from typing import Any


def run_activation_patch(
    model: Any,
    prompt_source: str,
    prompt_target: str,
    layer_name: str,
    token_position: int,
    target_token: str | None = None,
) -> dict[str, Any]:
    """Patch one source activation into a target prompt forward pass.

    Real patching is implemented for the project `LoadedModel` wrapper when it
    uses TransformerLens. Other backends return a simulated diagnostic that
    records the selected activation difference without changing logits.
    """

    if getattr(model, "backend", None) == "transformer_lens":
        try:
            return _run_transformer_lens_patch(
                model,
                prompt_source,
                prompt_target,
                layer_name,
                token_position,
                target_token,
            )
        except Exception as error:
            simulated = _run_simulated_patch(
                model,
                prompt_source,
                prompt_target,
                layer_name,
                token_position,
            )
            simulated["patching_error"] = str(error)
            return simulated

    return _run_simulated_patch(
        model,
        prompt_source,
        prompt_target,
        layer_name,
        token_position,
    )


def run_multi_site_activation_patch(
    model: Any,
    prompt_source: str,
    prompt_target: str,
    layer_name: str,
    token_positions: list[int],
    max_positions: int = 5,
    target_token: str | None = None,
) -> dict[str, Any]:
    """Run single-site patching across several token positions and summarize effects."""

    source_tokens, target_tokens = _prompt_tokens(model, prompt_source, prompt_target)
    selected_positions = token_positions[:max_positions]
    position_results = []

    for token_position in selected_positions:
        patch_result = run_activation_patch(
            model,
            prompt_source,
            prompt_target,
            layer_name,
            token_position,
            target_token=target_token,
        )
        largest_change = _largest_logit_change(patch_result)
        position_results.append(
            {
                "token_position": token_position,
                "source_token": _token_at(source_tokens, token_position),
                "target_token": _token_at(target_tokens, token_position),
                "patching_mode": patch_result.get("patching_mode", "unknown"),
                "patched": patch_result.get("patched", False),
                "largest_logit_shift_token": largest_change.get("token"),
                "largest_logit_shift_delta": largest_change.get("delta", 0.0),
                "top_logit_changes": patch_result.get("top_logit_changes", []),
                "activation_difference": patch_result.get("activation_difference"),
            }
        )

    largest_abs_shifts = [
        _patching_effect_size(result)
        for result in position_results
        if _patching_effect_size(result) is not None
    ]
    most_sensitive = None
    if position_results:
        most_sensitive = max(
            position_results,
            key=lambda item: _patching_effect_size(item) or 0.0,
        )["token_position"]

    aggregate_summary = {
        "positions_tested": len(position_results),
        "mean_abs_largest_shift": (
            sum(largest_abs_shifts) / len(largest_abs_shifts)
            if largest_abs_shifts
            else 0.0
        ),
        "max_abs_largest_shift": max(largest_abs_shifts) if largest_abs_shifts else 0.0,
        "most_sensitive_position": most_sensitive,
    }

    return {
        "layer": layer_name,
        "source_prompt": prompt_source,
        "target_prompt": prompt_target,
        "max_positions": max_positions,
        "positions": position_results,
        "aggregate_summary": aggregate_summary,
    }


def run_random_baseline_patch(
    model: Any,
    prompt_source: str,
    prompt_target: str,
    layer_name: str,
    candidate_token_count: int,
    exclude_positions: list[int] | None = None,
    samples: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Patch random same-layer token positions as a simple null baseline."""

    excluded = set(exclude_positions or [])
    candidate_positions = [
        position
        for position in range(candidate_token_count)
        if position not in excluded
    ]

    if not candidate_positions:
        return {
            "baseline_type": "random_same_layer_positions",
            "status": "skipped",
            "reason": "No candidate token positions remain after excluding top changed positions.",
            "samples": samples,
            "seed": seed,
            "positions_tested": [],
            "mean_abs_largest_shift": None,
            "max_abs_largest_shift": None,
            "top_changed_mean_abs_largest_shift": None,
            "effect_ratio": None,
            "interpretation": "Baseline could not run, so specificity cannot be estimated.",
        }

    rng = random.Random(seed)
    sampled_positions = rng.sample(
        candidate_positions,
        k=min(samples, len(candidate_positions)),
    )
    position_results = []

    for token_position in sampled_positions:
        patch_result = run_activation_patch(
            model,
            prompt_source,
            prompt_target,
            layer_name,
            token_position,
        )
        largest_change = _largest_logit_change(patch_result)
        position_results.append(
            {
                "token_position": token_position,
                "patching_mode": patch_result.get("patching_mode", "unknown"),
                "patched": patch_result.get("patched", False),
                "largest_logit_shift_token": largest_change.get("token"),
                "largest_logit_shift_delta": largest_change.get("delta", 0.0),
                "top_logit_changes": patch_result.get("top_logit_changes", []),
                "activation_difference": patch_result.get("activation_difference"),
            }
        )

    abs_shifts = [
        _patching_effect_size(result)
        for result in position_results
        if _patching_effect_size(result) is not None
    ]

    return {
        "baseline_type": "random_same_layer_positions",
        "status": "completed",
        "samples": samples,
        "seed": seed,
        "positions_tested": position_results,
        "mean_abs_largest_shift": (
            sum(abs_shifts) / len(abs_shifts)
            if abs_shifts
            else 0.0
        ),
        "max_abs_largest_shift": max(abs_shifts) if abs_shifts else 0.0,
        "top_changed_mean_abs_largest_shift": None,
        "effect_ratio": None,
        "interpretation": "Baseline completed; compare effect_ratio after top changed summary is attached.",
    }


def _run_transformer_lens_patch(
    loaded_model: Any,
    prompt_source: str,
    prompt_target: str,
    layer_name: str,
    token_position: int,
    target_token: str | None,
) -> dict[str, Any]:
    import torch

    tl_model = loaded_model.model
    hook_name = _layer_to_hook_name(layer_name)

    source_tokens = tl_model.to_tokens(prompt_source)
    target_tokens = tl_model.to_tokens(prompt_target)

    if token_position >= source_tokens.shape[1] or token_position >= target_tokens.shape[1]:
        raise ValueError(
            f"Token position {token_position} is outside the source or target prompt length."
        )

    _, source_cache = tl_model.run_with_cache(source_tokens)
    source_activation = source_cache[hook_name][0, token_position, :].detach()

    before_logits = tl_model(target_tokens)

    def patch_hook(activation: Any, hook: Any) -> Any:
        patched_activation = activation.clone()
        patched_activation[0, token_position, :] = source_activation.to(
            device=activation.device,
            dtype=activation.dtype,
        )
        return patched_activation

    after_logits = tl_model.run_with_hooks(
        target_tokens,
        fwd_hooks=[(hook_name, patch_hook)],
    )

    before_distribution = before_logits[0, token_position, :].detach().cpu()
    after_distribution = after_logits[0, token_position, :].detach().cpu()
    delta = after_distribution - before_distribution
    top_indices = torch.topk(delta.abs(), k=min(5, delta.numel())).indices.tolist()

    top_logit_changes = [
        {
            "token": _decode_token(tl_model, token_id),
            "before": float(before_distribution[token_id]),
            "after": float(after_distribution[token_id]),
            "delta": float(delta[token_id]),
        }
        for token_id in top_indices
    ]

    if target_token:
        target_ids = tl_model.to_tokens(target_token, prepend_bos=False)[0]
        if len(target_ids) > 0:
            token_id = int(target_ids[0])
            top_logit_changes.append(
                {
                    "token": _decode_token(tl_model, token_id),
                    "before": float(before_distribution[token_id]),
                    "after": float(after_distribution[token_id]),
                    "delta": float(delta[token_id]),
                }
            )

    largest = top_logit_changes[0] if top_logit_changes else None
    if largest:
        summary = (
            f"Patched {hook_name} at position {token_position}; largest logit shift "
            f"was {largest['token']!r} with delta {largest['delta']:.6f}."
        )
    else:
        summary = f"Patched {hook_name} at position {token_position}; no logit changes recorded."

    return {
        "layer": layer_name,
        "token_position": token_position,
        "source_prompt": prompt_source,
        "target_prompt": prompt_target,
        "patched": True,
        "patching_mode": "transformer_lens",
        "logit_shift_summary": summary,
        "top_logit_changes": top_logit_changes,
    }


def _run_simulated_patch(
    model: Any,
    prompt_source: str,
    prompt_target: str,
    layer_name: str,
    token_position: int,
) -> dict[str, Any]:
    source_result = model.run_prompt(prompt_source)
    target_result = model.run_prompt(prompt_target)

    source_activation = source_result.get("activations", {}).get(layer_name)
    target_activation = target_result.get("activations", {}).get(layer_name)
    activation_difference = None

    if source_activation is not None and target_activation is not None:
        source_vector = _position_vector(source_activation, token_position)
        target_vector = _position_vector(target_activation, token_position)
        if source_vector and target_vector:
            count = min(len(source_vector), len(target_vector))
            activation_difference = sum(
                abs(source_vector[index] - target_vector[index])
                for index in range(count)
            ) / count

    return {
        "layer": layer_name,
        "token_position": token_position,
        "source_prompt": prompt_source,
        "target_prompt": prompt_target,
        "patched": False,
        "patching_mode": "simulated",
        "activation_difference": activation_difference,
        "logit_shift_summary": (
            "Simulated patching mode recorded the selected activation difference; "
            "no target forward pass was patched."
        ),
        "top_logit_changes": [],
    }


def _layer_to_hook_name(layer_name: str) -> str:
    layer_index = int(layer_name.rsplit("_", 1)[1])
    return f"blocks.{layer_index}.hook_resid_post"


def _decode_token(model: Any, token_id: int) -> str:
    if getattr(model, "tokenizer", None) is not None:
        return model.tokenizer.decode([token_id])
    return str(token_id)


def _position_vector(value: Any, token_position: int) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if len(value.shape) == 3:
            value = value[0]
        if token_position >= len(value):
            return []
        return [float(item) for item in value[token_position].reshape(-1).tolist()]

    if isinstance(value, list):
        if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
            value = value[0]
        if token_position >= len(value):
            return []
        row = value[token_position]
        if isinstance(row, list):
            return [float(item) for item in row]
        return [float(row)]

    return []


def _prompt_tokens(model: Any, prompt_source: str, prompt_target: str) -> tuple[list[str], list[str]]:
    try:
        source_result = model.run_prompt(prompt_source)
        target_result = model.run_prompt(prompt_target)
        return list(source_result.get("tokens", [])), list(target_result.get("tokens", []))
    except Exception:
        return [], []


def _token_at(tokens: list[str], token_position: int) -> str | None:
    if token_position < len(tokens):
        return tokens[token_position]
    return None


def _largest_logit_change(patch_result: dict[str, Any]) -> dict[str, Any]:
    top_changes = patch_result.get("top_logit_changes") or []
    if not top_changes:
        return {"token": None, "delta": 0.0}
    return max(top_changes, key=lambda item: abs(item.get("delta", 0.0)))


def _patching_effect_size(result: dict[str, Any]) -> float | None:
    logit_delta = result.get("largest_logit_shift_delta")
    if logit_delta not in (None, 0):
        return abs(logit_delta)

    activation_difference = result.get("activation_difference")
    if activation_difference is not None:
        return abs(activation_difference)

    return 0.0 if logit_delta == 0 else None
