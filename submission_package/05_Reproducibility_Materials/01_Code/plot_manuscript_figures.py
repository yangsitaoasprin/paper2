"""
Generate manuscript-ready Figures 1-3 and Supplementary Figure S1
for the submission package.

Outputs are written to:
    submission_package/03_Figures/

Figure CSV inputs are read from:
    05_Reproducibility_Materials/04_Figure_Data/
"""

from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path

from bundle_paths import FIGURE_DATA_DIR, MANUSCRIPT_FIGURE_DIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter


DATA_DIR = FIGURE_DATA_DIR
OUTPUT_DIR = MANUSCRIPT_FIGURE_DIR

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10.2,
        "axes.labelsize": 11,
        "axes.titlesize": 12.0,
        "legend.fontsize": 9.6,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def load_csv_rows(file_path: Path) -> list[dict[str, str]]:
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_supported_render_stack() -> None:
    if platform.system() == "Windows" and matplotlib.__version__.startswith("3.11.0"):
        raise RuntimeError(
            "This Windows environment uses matplotlib 3.11.0, which was observed to crash "
            "during figure rendering in local verification. Please install matplotlib==3.10.6 "
            "or recreate the bundle environment from environment.yml before running "
            "plot_manuscript_figures.py."
        )


def save_figure(fig: plt.Figure, stem: str) -> None:
    ensure_supported_render_stack()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=400)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def soften_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def add_rect_patch(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
) -> None:
    patch = Rectangle(
        xy,
        width,
        height,
        linewidth=linewidth,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_artist(patch)


def add_flow_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    exp_text: str,
    detail: str,
    facecolor: str,
    edgecolor: str = "#355c7d",
    title_color: str = "#18324a",
    title_fontsize: float = 10.2,
    exp_fontsize: float = 9.2,
    detail_fontsize: float = 8.5,
) -> None:
    x, y = xy
    add_rect_patch(ax, (x, y), width, height, facecolor, edgecolor, 1.2)
    ax.text(x + width / 2, y + height * 0.78, title, ha="center", va="center", fontsize=title_fontsize, fontweight="bold", color=title_color)
    ax.text(x + width / 2, y + height * 0.55, exp_text, ha="center", va="center", fontsize=exp_fontsize, color="#2f3e46")
    ax.text(x + width / 2, y + height * 0.25, detail, ha="center", va="center", fontsize=detail_fontsize, color="#37474f", wrap=True)


def add_flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#5c6770",
        connectionstyle="arc3,rad=0.0",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_artist(arrow)


def add_elbow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    corner: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.plot(
        [start[0], corner[0], end[0]],
        [start[1], corner[1], end[1]],
        color="#5c6770",
        linewidth=1.2,
        solid_capstyle="round",
        zorder=1,
    )
    arrow = FancyArrowPatch(
        corner,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=0,
        color="#5c6770",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_artist(arrow)


def add_polyline_arrow(ax: plt.Axes, points: list[tuple[float, float]]) -> None:
    if len(points) < 2:
        return
    if len(points) > 2:
        for p0, p1 in zip(points[:-2], points[1:-1]):
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="#5c6770",
                linewidth=1.2,
                solid_capstyle="round",
                zorder=1,
            )
    arrow = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#5c6770",
        connectionstyle="arc3,rad=0.0",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_artist(arrow)


def add_architecture_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    exp_text: str,
    modules: list[str],
    role_text: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    x, y = xy
    add_rect_patch(ax, (x, y), width, height, facecolor, edgecolor, 1.2)
    ax.text(x + width / 2, y + height * 0.90, title, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#18324a")
    ax.text(x + width / 2, y + height * 0.835, exp_text, ha="center", va="center", fontsize=8.6, color="#355c7d")

    module_top = y + height * 0.71
    module_height = height * 0.098
    module_gap = height * 0.032
    for idx, module_text in enumerate(modules):
        box_y = module_top - idx * (module_height + module_gap)
        add_rect_patch(ax, (x + width * 0.07, box_y), width * 0.86, module_height, "#fbfcfd", "#c8d2dc", 0.9)
        ax.text(
            x + width / 2,
            box_y + module_height / 2,
            module_text,
            ha="center",
            va="center",
            fontsize=8.15,
            color="#2f3e46",
        )

    ax.text(
        x + width / 2,
        y + height * 0.10,
        role_text,
        ha="center",
        va="center",
        fontsize=7.9,
        color="#37474f",
        wrap=True,
    )


def plot_figure1() -> None:
    fig, ax = plt.subplots(figsize=(15.4, 7.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_rect_patch(ax, (0.28, 0.84), 0.44, 0.11, "#eef5fb", "#4c78a8", 1.2)
    ax.text(0.50, 0.895, "Curated benchmark: 15,005 drug-enzyme pairs", ha="center", va="center", fontsize=11.2, fontweight="bold", color="#16324f")
    ax.text(0.50, 0.855, "6 major human CYP450 isoforms", ha="center", va="center", fontsize=9.8, color="#355c7d")

    boxes = [
        {
            "xy": (0.05, 0.585),
            "width": 0.245,
            "title": "Baseline comparison",
            "exp": "Exp1 · Exp2 · Exp6",
            "detail": "How strong is a lightweight baseline?",
            "facecolor": "#eef5fb",
            "edge": "#4c78a8",
        },
        {
            "xy": (0.375, 0.585),
            "width": 0.245,
            "title": "Lightweight enzyme-information recovery",
            "exp": "Exp7-plus",
            "detail": "How much signal returns with minimal enzyme identity?",
            "facecolor": "#eef8f1",
            "edge": "#4d9078",
            "detail_fontsize": 8.3,
        },
        {
            "xy": (0.70, 0.585),
            "width": 0.245,
            "title": "Engineering variants on the Two-Tower line",
            "exp": "Exp8-11",
            "detail": "Do added modules yield stable gains?",
            "facecolor": "#fff3eb",
            "edge": "#dd8452",
            "title_fontsize": 9.7,
            "detail_fontsize": 8.3,
        },
        {
            "xy": (0.19, 0.325),
            "width": 0.29,
            "title": "Targeted enzyme-side refinement",
            "exp": "Exp13",
            "detail": "Can enzyme-side structure help when optimization is focused?",
            "facecolor": "#eef8f1",
            "edge": "#4d9078",
            "title_fontsize": 9.8,
            "detail_fontsize": 8.3,
        },
        {
            "xy": (0.53, 0.325),
            "width": 0.29,
            "title": "Scaffold robustness",
            "exp": "Exp12",
            "detail": "Do conclusions survive stricter structural generalization?",
            "facecolor": "#eef5fb",
            "edge": "#4c78a8",
            "detail_fontsize": 8.3,
        },
    ]

    default_width = 0.245
    height = 0.155
    for box in boxes:
        box_width = box.get("width", default_width)
        add_flow_box(
            ax,
            box["xy"],
            box_width,
            height,
            box["title"],
            box["exp"],
            box["detail"],
            box["facecolor"],
            edgecolor=box["edge"],
            title_fontsize=box.get("title_fontsize", 10.2),
            exp_fontsize=box.get("exp_fontsize", 9.2),
            detail_fontsize=box.get("detail_fontsize", 8.5),
        )

    for left_box, right_box in zip(boxes[:2], boxes[1:3]):
        left_width = left_box.get("width", default_width)
        end = (right_box["xy"][0], right_box["xy"][1] + height / 2)
        start = (left_box["xy"][0] + left_width, left_box["xy"][1] + height / 2)
        add_flow_arrow(ax, start, end)

    add_flow_arrow(ax, (0.50, 0.83), (0.18, 0.71))
    add_flow_arrow(ax, (0.50, 0.83), (0.49, 0.71))
    add_flow_arrow(ax, (0.50, 0.83), (0.80, 0.71))

    # Main-line transition from engineering into targeted refinement and scaffold robustness
    engineering = boxes[2]
    targeted = boxes[3]
    scaffold = boxes[4]
    engineering_width = engineering.get("width", default_width)
    targeted_width = targeted.get("width", default_width)

    add_polyline_arrow(
        ax,
        [
            (engineering["xy"][0] + engineering_width / 2, engineering["xy"][1]),
            (engineering["xy"][0] + engineering_width / 2, 0.505),
            (targeted["xy"][0] + targeted_width / 2, 0.505),
            (targeted["xy"][0] + targeted_width / 2, targeted["xy"][1] + height),
        ],
    )
    add_polyline_arrow(
        ax,
        [
            (targeted["xy"][0] + targeted_width, targeted["xy"][1] + height / 2),
            (scaffold["xy"][0], scaffold["xy"][1] + height / 2),
        ],
    )

    add_rect_patch(ax, (0.18, 0.045), 0.64, 0.125, "#edf7ee", "#4d9078", 1.3)
    ax.text(0.50, 0.125, "Design takeaway", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#1d4d3d")
    ax.text(
        0.50,
        0.075,
        "Task-aligned complexity is more defensible than indiscriminate complexity",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#1d4d3d",
    )

    add_flow_arrow(ax, (targeted["xy"][0] + targeted_width / 2, targeted["xy"][1]), (0.42, 0.17))
    add_flow_arrow(ax, (scaffold["xy"][0] + scaffold.get("width", default_width) / 2, scaffold["xy"][1]), (0.60, 0.17))

    fig.suptitle("Figure 1. Main-line experiment map of the study", fontsize=12.6, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92)
    save_figure(fig, "figure1_mainline_experiment_map")


def plot_figure2() -> None:
    rows = load_csv_rows(DATA_DIR / "figure2_cnn_ablation.csv")
    x = [int(row["depth_index"]) for row in rows]
    x_labels = [row["depth_label"] for row in rows]
    auc = [float(row["test_auc"]) for row in rows]
    pr_auc = [float(row["test_pr_auc"]) for row in rows]
    mcc = [float(row["test_mcc"]) for row in rows]
    params = [int(row["parameters"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.9, 5.75))
    ax.plot(x, auc, marker="o", linewidth=2.4, color="#1f4e79", label="Test AUROC")
    ax.plot(x, pr_auc, marker="s", linewidth=2.2, color="#2a9d8f", label="Test AUPRC")
    ax.plot(x, mcc, marker="^", linewidth=2.2, color="#e76f51", label="Test MCC")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Enzyme encoder depth")
    ax.set_ylabel("Performance")
    ax.set_ylim(0.52, 0.94)
    ax.margins(x=0.06)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    soften_axes(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.985, 0.025), borderaxespad=0.0)

    for xi, param in zip(x, params):
        ax.text(
            xi,
            0.532,
            f"{param / 1_000_000:.2f}M" if param >= 1_000_000 else f"{round(param/1000)}k",
            ha="center",
            va="bottom",
            fontsize=8.9,
            color="#666666",
        )

    ax.annotate(
        "Most balanced variant",
        xy=(2, pr_auc[1]),
        xytext=(1.58, 0.886),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=9.0,
    )
    ax.annotate(
        "Higher MCC,\nweaker ranking metrics",
        xy=(3, mcc[2]),
        xytext=(2.15, 0.548),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#333333"},
        fontsize=9.0,
    )

    fig.suptitle("Figure 2. Enzyme-side CNN Depth Ablation Under Controlled Evaluation (Exp13)", fontsize=11.8)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.14, top=0.90)
    save_figure(fig, "figure2_enzyme_cnn_ablation")


def plot_figure3() -> None:
    rows = load_csv_rows(DATA_DIR / "figure3_scaffold_robustness.csv")
    ratio_order = ["80/10/10", "70/15/15", "60/20/20", "50/25/25"]
    ratio_x = list(range(len(ratio_order)))
    model_order = ["Drug-Only", "Morgan-CNN", "Exp7-plus", "Two-Tower V1", "Exp9", "CNN-2"]
    colors = {
        "Drug-Only": "#7a7a7a",
        "Morgan-CNN": "#1f4e79",
        "Exp7-plus": "#2a9d8f",
        "Two-Tower V1": "#f4a261",
        "Exp9": "#d1495b",
        "CNN-2": "#7b6fd0",
    }

    series: dict[str, dict[str, float]] = {}
    for row in rows:
        model = row["model"]
        series.setdefault(model, {})
        series[model][row["scaffold_ratio"]] = float(row["test_auc"])

    fig, ax = plt.subplots(figsize=(9.8, 5.95))
    for model in model_order:
        ratio_map = series[model]
        y = [ratio_map[ratio] for ratio in ratio_order]
        ax.plot(
            ratio_x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=model,
            color=colors.get(model, "#333333"),
        )

    ax.set_xticks(ratio_x)
    ax.set_xticklabels(ratio_order)
    ax.set_xlabel("Scaffold split ratio")
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.74, 0.83)
    ax.set_xlim(-0.15, 3.15)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    soften_axes(ax)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.115),
    )

    ax.annotate(
        "Best single point:\nTwo-Tower V1, AUROC 0.827",
        xy=(0, series["Two-Tower V1"]["80/10/10"]),
        xytext=(0.52, 0.8253),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=8.9,
    )
    ax.text(
        1.88,
        0.7478,
        "Scaffold-leading group:\nCNN-2 / Two-Tower V1 / Exp9",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fbfbfb", "edgecolor": "#d0d0d0", "alpha": 0.95},
    )

    fig.suptitle("Figure 3. Scaffold Robustness Profile Across Four Scaffold Ratios (Exp12)", fontsize=11.9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.24)
    save_figure(fig, "figure3_scaffold_robustness_profile")


def plot_figure_s1() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 7.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_rect_patch(ax, (0.20, 0.88), 0.60, 0.08, "#eef5fb", "#4c78a8", 1.2)
    ax.text(0.50, 0.925, "Supplementary Figure S1. Representative Model-Route Schematics", ha="center", va="center", fontsize=12.2, fontweight="bold", color="#16324f")
    ax.text(
        0.50,
        0.892,
        "Reviewer-oriented structural guide to the current manuscript line",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#355c7d",
    )

    cards = [
        {
            "xy": (0.035, 0.50),
            "title": "Drug-Only",
            "exp": "Exp6",
            "modules": [
                "Drug: Morgan fingerprint (2048)",
                "Encoder: compact MLP",
                "Enzyme branch: none",
            ],
            "role": "Structure-only lower bound.",
            "face": "#f2f2f2",
            "edge": "#7a7a7a",
        },
        {
            "xy": (0.278, 0.50),
            "title": "Morgan-CNN",
            "exp": "Exp1",
            "modules": [
                "Drug: Morgan fingerprint (2048)",
                "Enzyme: tokenized sequence",
                "Fusion: sequence CNN + compact head",
            ],
            "role": "Strong released-split lightweight baseline.",
            "face": "#eef5fb",
            "edge": "#4c78a8",
        },
        {
            "xy": (0.521, 0.50),
            "title": "Lightweight Enzyme-ID Recovery",
            "exp": "Exp7-plus",
            "modules": [
                "Drug: Morgan + 8 descriptors",
                "Enzyme: ID embedding only",
                "Fusion: 160-dim joint classifier",
            ],
            "role": "Minimal enzyme ID recovers much of the signal.",
            "face": "#eef8f1",
            "edge": "#4d9078",
        },
        {
            "xy": (0.145, 0.14),
            "title": "Two-Tower V1",
            "exp": "Exp2",
            "modules": [
                "Drug tower: 2056 -> 32",
                "Enzyme tower: sequence CNN + ID",
                "Predictor: 64-dim joint MLP",
            ],
            "role": "Reference explicit dual-branch route.",
            "face": "#fff3eb",
            "edge": "#dd8452",
        },
        {
            "xy": (0.505, 0.14),
            "title": "CNN-2 Targeted Refinement",
            "exp": "Exp13",
            "modules": [
                "Drug tower: fixed Exp13 route",
                "Enzyme branch: 2-layer CNN",
                "Scope: targeted enzyme-side depth change",
            ],
            "role": "Focused enzyme-side refinement over broad expansion.",
            "face": "#f3effd",
            "edge": "#7b6fd0",
        },
    ]

    card_width = 0.215
    card_height = 0.29
    for card in cards:
        add_architecture_card(
            ax,
            card["xy"],
            card_width,
            card_height,
            card["title"],
            card["exp"],
            card["modules"],
            card["role"],
            card["face"],
            card["edge"],
        )

    add_rect_patch(ax, (0.11, 0.028), 0.78, 0.062, "#fbfbfb", "#cfd7de", 1.1)
    ax.text(
        0.50,
        0.058,
        "Exp8-11 remain localized modifications on the Two-Tower V1 line, so they are summarized textually rather than redrawn as separate full architectures.",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#4a5560",
        wrap=True,
    )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.95)
    save_figure(fig, "figureS1_representative_model_routes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript figure assets from CSV inputs.")
    parser.add_argument(
        "--figures",
        nargs="*",
        choices=["1", "2", "3", "S1"],
        default=["1", "2", "3"],
        help="Select which figures to generate. Default: 1 2 3",
    )
    args = parser.parse_args()

    if "1" in args.figures:
        plot_figure1()
    if "2" in args.figures:
        plot_figure2()
    if "3" in args.figures:
        plot_figure3()
    if "S1" in args.figures:
        plot_figure_s1()


if __name__ == "__main__":
    main()
