"""Small visual outputs for prompt comparison reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_layer_difference_chart(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a bar chart of mean activation differences by layer."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_differences = report.get("layer_differences", [])
    layers = [item["layer"].replace("layer_", "L") for item in layer_differences]
    means = [item["mean_activation_difference"] for item in layer_differences]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(layers, means, color="#2f6f73")
    ax.set_title("Mean Activation Difference by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean absolute difference")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def write_top_layer_token_heatmap(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a compact heatmap for token differences in the most changed layer."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    top_layer = (report.get("top_changed_tokens") or [{}])[0]
    token_rows = top_layer.get("tokens", [])
    sorted_tokens = sorted(token_rows, key=lambda item: item["position"])
    labels = [_shorten_token_label(item["token"]) for item in sorted_tokens]
    values = [item["difference"] for item in sorted_tokens]

    if not values:
        labels = ["no comparable tokens"]
        values = [0.0]

    fig, ax = plt.subplots(figsize=(12, 3))
    image = ax.imshow([values], aspect="auto", cmap="viridis")

    ax.set_title(f"Top Token Differences in {top_layer.get('layer', 'top layer')}")
    ax.set_yticks([0])
    ax.set_yticklabels(["difference"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Aligned token position")

    for index, value in enumerate(values):
        ax.text(index, 0, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean absolute difference")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def write_multi_site_patching_chart(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a bar chart of patching effect per tested token position."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    patching = report.get("multi_site_activation_patching") or {}
    positions = patching.get("positions", [])
    labels = [str(item["token_position"]) for item in positions]
    values = [_patching_plot_value(item) for item in positions]
    ylabel = _patching_y_label(positions)

    if not values:
        labels = ["none"]
        values = [0.0]
    y_max = max(values) if values else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color="#865d3c")
    ax.set_title("Multi-Site Patching: Largest Shift")
    ax.set_xlabel("Token position")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, y_max * 1.2 if y_max > 0 else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def write_patching_baseline_comparison_chart(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a compact comparison of top changed patching vs random baseline."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    baseline = report.get("random_baseline_patching") or {}
    top_mean = baseline.get("top_changed_mean_abs_largest_shift") or 0.0
    random_mean = baseline.get("mean_abs_largest_shift") or 0.0
    values = [top_mean, random_mean]
    y_max = max(values)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        ["top changed", "random baseline"],
        values,
        color=["#2f6f73", "#865d3c"],
    )
    ax.set_title("Patching Baseline Comparison")
    ax.set_ylabel(_baseline_y_label(baseline))
    ax.set_ylim(0, y_max * 1.2 if y_max > 0 else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def _shorten_token_label(token: str, max_length: int = 18) -> str:
    clean = token.replace("\n", "\\n")
    if len(clean) <= max_length:
        return clean
    return f"{clean[: max_length - 3]}..."


def _patching_plot_value(position: dict[str, Any]) -> float:
    logit_delta = position.get("largest_logit_shift_delta")
    if logit_delta not in (None, 0):
        return abs(logit_delta)
    activation_difference = position.get("activation_difference")
    if activation_difference is not None:
        return abs(activation_difference)
    return 0.0


def _patching_y_label(positions: list[dict[str, Any]]) -> str:
    if positions and all(not item.get("patched") for item in positions):
        return "Mean activation difference (simulated fallback)"
    return "Largest absolute logit shift"


def _baseline_y_label(baseline: dict[str, Any]) -> str:
    positions = baseline.get("positions_tested") or []
    if positions and all(not item.get("patched") for item in positions):
        return "Mean activation difference (simulated fallback)"
    return "Mean largest absolute logit shift"
