"""Render the fixed Spec2Behavior loss and accuracy curves from verified runs."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ASSETS = Path(__file__).resolve().parents[1] / "assets/epoch-sweep"
COLLECTIONS = {
    "model-spec-original": ("Model Spec · original", "#24629F", "o"),
    "model-spec-generic": ("Model Spec · generic", "#A2590D", "s"),
    "constitution-original": ("Constitution · original", "#167658", "^"),
    "constitution-generic": ("Constitution · generic", "#8B4D85", "D"),
}
BATCHES = {
    32: ("#24629F", "o"),
    16: ("#A2590D", "s"),
    8: ("#167658", "^"),
}
TESTS = {
    "model-spec-original": ("OpenAI · Model Spec", (77, 85), [78, 80, 82, 84]),
    "model-spec-generic": ("Model Spec · generic", (90, 99), [90, 92, 94, 96, 98]),
    "constitution-original": ("Constitution · original", (80, 95), [80, 85, 90, 95]),
    "constitution-generic": ("Constitution · generic", (68, 88), [68, 72, 76, 80, 84, 88]),
}
HORIZONS = (0, 1, 3, 5, 10)
INK = "#18252F"
MUTED = "#536270"


def load_loss():
    with (ASSETS / "training-loss.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = {(row["model"], int(row["epoch"])): float(row["online_example_mean_nll"]) for row in rows}
    if len(rows) != 120 or len(points) != 120:
        raise ValueError("Expected 120 distinct published epoch means")
    return points


def load_accuracy():
    scores, totals = {}, {}
    with (ASSETS / "evaluation-accuracy.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            test, model = row["benchmark"], row["model"]
            count, passed = int(row["total"]), int(row["passed"])
            if count <= 0 or not 0 <= passed <= count or not math.isclose(passed/count, float(row["pass_rate"])):
                raise ValueError(f"Invalid published score for {test}/{model}")
            if (test, model) in scores or totals.setdefault(test, count) != count:
                raise ValueError(f"Duplicate score or inconsistent denominator for {test}/{model}")
            scores[test, model] = 100 * passed / count
    if len(scores) != 196 or set(totals) != set(TESTS):
        raise ValueError("Expected all 196 published evaluation cells")
    return scores, totals


def frame(fig, title, subtitle):
    fig.text(.065, .967, title, fontsize=23, weight="bold", color=INK, va="top")
    fig.text(.065, .925, subtitle, fontsize=11.2, color=MUTED, va="top")


def legend(fig, handles, y):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.065, y),
               bbox_transform=fig.transFigure, ncol=len(handles), frameon=False,
               borderaxespad=0, columnspacing=1.6, fontsize=10.5)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DCE4EA", linewidth=.75)
    ax.tick_params(axis="both", length=3.5, color="#9AAAB5", labelcolor=MUTED)


def loss_figure(points):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.97, bottom=.13, top=.80, wspace=.15, hspace=.28)
    frame(fig, "Training loss by epoch", "Spec2Behavior  /  12 configurations on the ten-epoch schedule")
    legend(fig, [Line2D([], [], color=color, marker=marker, lw=2, label=f"Batch {bs}")
                 for bs, (color, marker) in BATCHES.items()], .886)
    for ax, (collection, (label, _, _)) in zip(axes.flat, COLLECTIONS.items(), strict=True):
        style_axis(ax)
        ax.set_title(label, loc="left", fontsize=13, weight="semibold", pad=8)
        for bs, (color, marker) in BATCHES.items():
            vals = [points[f"{collection}-b{bs}", epoch] for epoch in range(1, 11)]
            ax.plot(range(1, 11), vals, color=color, marker=marker, ms=4.7, lw=2)
            ax.annotate(f"{vals[-1]:.3f}", (10, vals[-1]), xytext=(7, 3),
                        textcoords="offset points", fontsize=9.5, color=color, va="center")
        ax.set_xticks(range(1, 11))
        ax.set_xlim(.7, 11.05)
        ax.set_ylim(0, 2.08)
    fig.supxlabel("Epoch within the ten-epoch run", y=.077, fontsize=11, color=INK)
    fig.supylabel("Mean training NLL", x=.035, fontsize=11, color=INK)
    fig.text(.065, .037, "Per-epoch, example-weighted mean of pre-update training NLL. Labels show epoch 10.", fontsize=10, color=MUTED)
    return fig


def accuracy_figure(scores, totals):
    fig, axes = plt.subplots(4, 3, figsize=(14, 11.8), sharex=True, sharey="row")
    fig.subplots_adjust(left=.145, right=.975, bottom=.115, top=.823, wspace=.18, hspace=.24)
    frame(fig, "Evaluation accuracy by training horizon", "Spec2Behavior  /  Automatic rubric pass rate on four fixed policy tests")
    handles = [Line2D([], [], color=color, marker=marker, lw=1.5, ls="--", label=label)
               for label, color, marker in COLLECTIONS.values()]
    handles.append(Line2D([], [], color=INK, marker="d", lw=0, label="Base"))
    legend(fig, handles, .889)
    for j, bs in enumerate(BATCHES):
        axes[0, j].set_title(f"Batch {bs}", fontsize=13.5, weight="semibold", pad=10, color=INK)
    for i, (test, (label, limits, ticks)) in enumerate(TESTS.items()):
        for j, bs in enumerate(BATCHES):
            ax = axes[i, j]
            style_axis(ax)
            base = scores[test, "base"]
            ax.axhline(base, color="#A3ADB6", lw=.8, ls=":", zorder=1)
            for collection, (_, color, marker) in COLLECTIONS.items():
                vals = [base, *[scores[test, f"{collection}-b{bs}-e{e}"] for e in HORIZONS[1:]]]
                assert all(limits[0] <= value <= limits[1] for value in vals)
                ax.plot(HORIZONS, vals, marker=marker, markevery=[1, 2, 3, 4], ms=4.7,
                        ls="--", lw=1.55, color=color, alpha=.95, zorder=3)
            ax.scatter([0], [base], s=28, color=INK, marker="d", zorder=4)
            ax.set_xlim(-.35, 10.4)
            ax.set_ylim(*limits)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_xticks(HORIZONS)
        a = axes[i, 0].get_position()
        fig.text(.025, (a.y0 + a.y1)/2 + .014, label.replace(" · ", "\n"), fontsize=10.6,
                 color=INK, weight="semibold", va="center", linespacing=1.17)
        fig.text(.025, (a.y0 + a.y1)/2 - .033, f"n = {totals[test]}", fontsize=9.4, color=MUTED, va="center")
    fig.supxlabel("Training horizon (epochs)", y=.072, fontsize=11, color=INK)
    fig.text(.065, .042, "Points show the base and separate 1-, 3-, 5-, and 10-epoch runs. Dashes connect measured endpoints.", fontsize=9.9, color=MUTED)
    fig.text(.065, .021, "Each test row uses its own labelled y-range. Exact scores and paired intervals are in the evaluation report.", fontsize=9.9, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#A4B1BB", "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
        "pdf.fonttype": 42,
    })
    loss = load_loss()
    accuracy, totals = load_accuracy()
    figures = [(loss_figure(loss), "training-loss"), (accuracy_figure(accuracy, totals), "evaluation-accuracy")]
    with PdfPages(args.output / "loss-and-accuracy.pdf") as pdf:
        for fig, name in figures:
            fig.savefig(args.output / f"{name}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    print("Rendered the published loss and accuracy curves.")


if __name__ == "__main__":
    main()
