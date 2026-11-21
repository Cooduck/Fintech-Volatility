"""Build the report's vector figures from the frozen research archive."""

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter


FIGURES = Path(__file__).resolve().parent
RESULTS = FIGURES.parents[1] / "results" / "v5"
INK = "#162638"
MUTED = "#495a6b"
BLUE = "#205782"
TEAL = "#00796c"
ORANGE = "#9d4d13"
RULE = "#98a7b5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
})


def archive_rows(filename):
    with (RESULTS / filename).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(figure, name):
    figure.savefig(
        FIGURES / name,
        metadata={"Creator": "Fintech Volatility report", "CreationDate": None},
    )
    plt.close(figure)


def architecture():
    figure, ax = plt.subplots(figsize=(6.75, 3.65))
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    def box(x, y, width, title, detail, tone, background, height=14):
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.17,rounding_size=1.1",
            linewidth=0.75, edgecolor=tone, facecolor=background,
        ))
        ax.text(x, y + 2.3, title, ha="center", va="center", weight="bold", fontsize=9.2)
        ax.text(x, y - 2.6, detail, ha="center", va="center", fontsize=8.1, color=MUTED)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=MUTED,
        ))

    news = (BLUE, "#eef5fb")
    market = (TEAL, "#edf7f4")
    label = (ORANGE, "#fcf3ec")
    model = (MUTED, "#f3f5f8")

    box(17, 91, 30, "News archive", "Headlines, timestamps, leads", *news)
    box(50, 91, 30, "Market and disclosures", "Lagged bars, QQQ, VIX, SEC", *market)
    box(83, 91, 30, "Realized prices", "Opening gaps and session bars", *label)
    box(17, 71, 30, "Closed-book scoring", "Cache, timing rules, clusters", *news)
    box(50, 71, 30, "Baseline alignment", "Data known at the decision cutoff", *market)
    box(83, 71, 30, "Target construction", "B1 with gap; B2 regular hours", *label)
    box(33.5, 51, 63, "As-of feature assembly", "A: baseline     C: sentiment and context     D: full event features", *model)
    box(17, 30, 30, "Walk-forward models", "B1 classifier and B2 regressor", *model)
    box(50, 30, 30, "Earnings review", "Verified leads and event risk", *market)
    box(83, 30, 30, "Label use", "Completed training / test windows", *label)
    box(50, 9, 96, "Historical outputs and audit", "Rankings and alerts   |   Earnings review   |   Paired AUC, lead–lag tests, replay, and reproduction", *model)

    for center in (17, 50, 83):
        arrow((center, 83.8), (center, 78.2))
    for center in (17, 50):
        arrow((center, 63.8), (center, 58.2))
        arrow((center - 11.5, 43.8), (center - 11.5, 37.2))
    arrow((83, 63.8), (83, 37.2))
    ax.text(17, 40.4, "Pooled / regular day", va="center", ha="center",
            color=MUTED, fontsize=7.7)
    ax.text(50, 40.4, "Phase B earnings", va="center", ha="center",
            color=MUTED, fontsize=7.7)
    for center in (17, 50, 83):
        arrow((center, 22.8), (center, 16.2))
    save(figure, "system_architecture.pdf")


def replay():
    rows = archive_rows("event_replay.csv")
    figure, axes = plt.subplots(1, 2, figsize=(6.75, 2.36))
    figure.subplots_adjust(left=0.11, bottom=0.25, right=0.98, top=0.75, wspace=0.46)
    for ax, target, title, ylabel, axis_range, ticks in zip(
        axes, ("B1", "B2"), ("B1 · with opening gap", "B2 · regular hours"),
        ("Probability score", "Standardized score"),
        ((0.77, 0.83), (1.08, 1.70)), ((0.78, 0.80, 0.82), (1.10, 1.30, 1.50, 1.70)),
    ):
        sample = sorted((row for row in rows if row["target"] == target), key=lambda row: row["cut"])
        cutoffs = list(range(len(sample)))
        release, = (i for i, row in enumerate(sample) if row["official_arrival"] == "True")
        labels = [
            datetime.fromisoformat(row["cut"]).astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")
            for row in sample
        ]
        ax.axvline(release, color=RULE, lw=0.9, zorder=0)
        for field, color, marker, line, legend in (
            ("score", BLUE, "o", "-", "Raw model"),
            ("display_score", TEAL, "s", "--", "Event overlay"),
        ):
            ax.plot(cutoffs, [float(row[field]) for row in sample], line,
                    color=color, marker=marker, markersize=4.3, linewidth=1.45, label=legend)
        ax.set(
            title=title, ylabel=ylabel, xlim=(-0.22, cutoffs[-1] + 0.22),
            ylim=axis_range, yticks=ticks, xticks=cutoffs, xticklabels=labels,
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5eaf0", linewidth=0.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, 1.01))
    figure.supxlabel("Cutoff snapshot (ET)", y=0.04, fontsize=9)
    save(figure, "event_replay.pdf")


def clock_comparison():
    rows = archive_rows("phase_b/clock_matched_diagnostic.csv")
    figure, ax = plt.subplots(figsize=(6.75, 2.70))
    figure.subplots_adjust(left=0.32, bottom=0.2, right=0.97, top=0.95)
    rows_by_key = {(row["target"], row["contrast"]): row for row in rows}
    order = ("conservative_D-A", "bounded_D-A", "clock_difference")
    labels = ("Conservative gain", "Bounded gain", "Difference")
    yticks, tick_labels = [], []
    for target, positions in (("B1", (6, 5, 4)), ("B2", (2, 1, 0))):
        for contrast, y, label, color, marker in zip(
            order, positions, labels, (BLUE, TEAL, ORANGE), ("o", "s", "D"),
        ):
            row = rows_by_key[target, contrast]
            point, low, high = (float(row[column]) for column in ("estimate", "lo95", "hi95"))
            ax.errorbar(point, y, xerr=[[point - low], [high - point]],
                        fmt=marker, color=color, lw=1.45, capsize=3, markersize=4.7)
            yticks.append(y)
            tick_labels.append(f"{target}  {label}")
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(3, color="#e5eaf0", linewidth=0.7, zorder=0)
    ax.set(xlabel="Difference in ROC-AUC", xlim=(-0.024, 0.12), ylim=(-0.55, 6.55),
           xticks=[-0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
           yticks=yticks, yticklabels=tick_labels)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e5eaf0", linewidth=0.6)
    ax.set_axisbelow(True)
    save(figure, "clock_matched.pdf")


if __name__ == "__main__":
    architecture()
    replay()
    clock_comparison()
