"""Activation capture helpers for TransformerLens and Hugging Face models."""

from __future__ import annotations

from typing import Any


def capture_transformer_lens_activations(model: Any, prompt: str) -> dict[str, Any]:
    """Run a TransformerLens model and capture residual stream activations by layer."""

    _, cache = model.run_with_cache(prompt)
    activations: dict[str, Any] = {}

    for layer in range(model.cfg.n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key in cache:
            activations[f"layer_{layer}"] = cache[key].detach().cpu()

    return activations


def capture_huggingface_activations(tokenizer: Any, model: Any, prompt: str) -> dict[str, Any]:
    """Run a Hugging Face causal LM and capture hidden states by layer."""

    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    activations: dict[str, Any] = {}
    for index, hidden_state in enumerate(outputs.hidden_states[1:]):
        activations[f"layer_{index}"] = hidden_state.detach().cpu()

    return activations
