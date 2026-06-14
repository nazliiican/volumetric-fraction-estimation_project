from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import config

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def _save(fig, save_path):
    if save_path is None:
        return
    save_path = Path(save_path)
    if not save_path.is_absolute():
        config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = config.FIGURE_DIR / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")


def plot_material_distribution(targets, save_path=None):
    """Bar chart: how many mixtures contain each material."""
    materials = targets["materials"]
    counts = np.asarray(targets["presence"]).sum(axis=0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(materials, counts, color="#4C72B0")
    ax.set_xlabel("Material")
    ax.set_ylabel("Number of mixtures containing it")
    ax.set_title("Material frequency across mixtures")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    fig.tight_layout()
    _save(fig, save_path)
    return ax


def plot_mixture_size_distribution(targets, save_path=None):
    """Bar chart: how many mixtures have 1, 2 or 3 materials."""
    sizes = np.asarray(targets["presence"]).sum(axis=1).astype(int)
    values, counts = np.unique(sizes, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(v) for v in values], counts, color="#55A868")
    ax.set_xlabel("Number of materials in the mixture")
    ax.set_ylabel("Number of mixtures")
    ax.set_title("Mixture size distribution")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    fig.tight_layout()
    _save(fig, save_path)
    return ax


def plot_training_curve(history, save_path=None):

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="train loss", marker="o", ms=3)
    ax.plot(epochs, history["val_loss"], label="val loss", marker="o", ms=3)
    if "val_mae" in history:
        ax.plot(epochs, history["val_mae"], label="val MAE", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curve")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return ax


def plot_predicted_vs_true(y_true, y_pred, materials=None, save_path=None):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if materials is not None and yt.ndim == 2 and yt.shape[1] == len(materials):
        for i, m in enumerate(materials):
            ax.scatter(yt[:, i], yp[:, i], s=14, alpha=0.6, label=m)
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.scatter(yt.ravel(), yp.ravel(), s=14, alpha=0.5, color="#4C72B0")

    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("True fraction")
    ax.set_ylabel("Predicted fraction")
    ax.set_title("Predicted vs. true composition")
    fig.tight_layout()
    _save(fig, save_path)
    return ax


def plot_model_comparison(comparison_df, metric="MAE", title=None, ylabel=None,
                          value_fmt="{:.3f}", save_path=None):
    
    models = comparison_df["Model"].tolist()
    values = comparison_df[metric].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, values, color="#C44E52")
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    ax.set_title(title if title is not None else f"Model comparison: {metric}")
    ax.tick_params(axis="x", rotation=20)
    ax.margins(y=0.15)  # headroom so the top bar label is not clipped
    for i, v in enumerate(values):
        ax.text(i, v, value_fmt.format(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return ax


def plot_per_material_mae(per_material, materials=None, title=None, save_path=None):
    
    fig, ax = plt.subplots(figsize=(8, 4.5))

    first_value = next(iter(per_material.values()))
    if isinstance(first_value, dict):
        model_names = list(per_material.keys())
        mats = materials or list(first_value.keys())
        x = np.arange(len(mats))
        width = 0.8 / len(model_names)
        for k, name in enumerate(model_names):
            vals = [per_material[name][m] for m in mats]
            ax.bar(x + k * width, vals, width=width, label=name)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(mats)
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1),
                  frameon=False, title="Model")
    else:
        mats = materials or list(per_material.keys())
        vals = [per_material[m] for m in mats]
        ax.bar(mats, vals, color="#8172B3")

    ax.set_xlabel("Material")
    ax.set_ylabel("MAE")
    ax.set_title(title if title is not None else "Per-material MAE")
    fig.tight_layout()
    _save(fig, save_path)
    return ax


def plot_example_images(image_df, codes=None, n_codes=8, seed=42, save_path=None):
    
    from PIL import Image  # lazy import

    if codes is None:
        unique_codes = image_df["Code"].drop_duplicates().tolist()
        rng = np.random.default_rng(seed)
        codes = list(rng.choice(unique_codes, size=min(n_codes, len(unique_codes)),
                                replace=False))

    n = len(codes)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")
    for ax, code in zip(axes, codes):
        row = image_df[image_df["Code"] == code].iloc[0]
        img = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"Code {code}\n{row['Material']}", fontsize=9)

    fig.suptitle("Example mixture images", y=1.0)
    fig.tight_layout()
    _save(fig, save_path)
    return axes


def plot_example_predictions(
    image_df, codes, y_true, y_pred, materials, save_path=None
):
    
    from PIL import Image  # lazy import

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    n = len(codes)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3.2 * n))
    axes = np.atleast_1d(axes).ravel()

    for ax, code, t, p in zip(axes, codes, yt, yp):
        row = image_df[image_df["Code"] == code].iloc[0]
        img = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        def fmt(vec):
            return ", ".join(
                f"{m}:{v:.2f}" for m, v in zip(materials, vec) if v > 0.01
            )

        ax.set_title(
            f"Code {code}\ntrue   {fmt(t)}\npred  {fmt(p)}",
            fontsize=9,
            loc="left",
        )

    fig.tight_layout()
    _save(fig, save_path)
    return axes
