"""Model loading with TransformerLens preferred and Hugging Face as fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Any

from mechinterp_probe.activation_capture import (
    capture_huggingface_activations,
    capture_transformer_lens_activations,
)


HF_MODEL_NAMES = {
    "gpt2-small": "gpt2",
}


@dataclass
class LoadedModel:
    """A small wrapper around whichever model backend is available."""

    model_name: str
    backend: str
    model: Any
    tokenizer: Any | None = None

    def run_prompt(self, prompt: str) -> dict[str, Any]:
        if self.backend == "transformer_lens":
            activations = capture_transformer_lens_activations(self.model, prompt)
            tokens = self.model.to_str_tokens(prompt)
        elif self.backend == "huggingface":
            activations = capture_huggingface_activations(self.tokenizer, self.model, prompt)
            token_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        return {
            "model": self.model_name,
            "backend": self.backend,
            "prompt": prompt,
            "tokens": tokens,
            "activations": activations,
        }


def load_model(model_name: str = "gpt2-small") -> LoadedModel:
    """Load GPT-2 Small, preferring TransformerLens when installed."""

    try:
        with _without_local_pytest_shadow():
            from transformer_lens import HookedTransformer

            try:
                model = HookedTransformer.from_pretrained(model_name, local_files_only=True)
            except Exception:
                model = HookedTransformer.from_pretrained(model_name)
        model.eval()
        return LoadedModel(
            model_name=model_name,
            backend="transformer_lens",
            model=model,
        )
    except Exception as transformer_lens_error:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_name = HF_MODEL_NAMES.get(model_name, model_name)
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_name, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(hf_name, local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(hf_name)
                model = AutoModelForCausalLM.from_pretrained(hf_name)
            model.eval()
            return LoadedModel(
                model_name=model_name,
                backend="huggingface",
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as huggingface_error:
            raise RuntimeError(
                "Could not load the model with TransformerLens or Hugging Face. "
                "Install dependencies from requirements.txt and ensure model downloads are available."
            ) from huggingface_error


def run_prompt(prompt: str, model: LoadedModel | None = None) -> dict[str, Any]:
    """Run one prompt through a loaded model and return captured activations."""

    loaded_model = model or load_model()
    return loaded_model.run_prompt(prompt)


class _without_local_pytest_shadow:
    """Avoid repo-local pytest.py shadowing real pytest during TransformerLens import."""

    def __enter__(self) -> None:
        self.original_path = list(sys.path)
        self.original_pytest = sys.modules.get("pytest")
        shadow_paths = {
            str(Path(path or os.getcwd()).resolve())
            for path in sys.path
            if (Path(path or os.getcwd()).resolve() / "pytest.py").exists()
        }
        sys.path = [
            path
            for path in sys.path
            if str(Path(path or os.getcwd()).resolve()) not in shadow_paths
        ]
        loaded_pytest = sys.modules.get("pytest")
        loaded_file = getattr(loaded_pytest, "__file__", "") if loaded_pytest else ""
        if loaded_file and str(Path(loaded_file).resolve().parent) in shadow_paths:
            sys.modules.pop("pytest", None)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        sys.path = self.original_path
        if self.original_pytest is not None:
            sys.modules["pytest"] = self.original_pytest
