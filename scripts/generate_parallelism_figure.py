#!/usr/bin/env python3
"""Recreate paper-style DP / TP / PP schematic (forward only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"

# Light purple layers + warm peach data + soft sand for gpu-comm / weight update
C_LAYER = "#C9B6E4"
C_LAYER_EDGE = "#5C3D7A"
C_DATA = "#F0B27A"
C_DATA_EDGE = "#B86A2A"
C_COMM = "#EAD9B5"
C_COMM_EDGE = "#9A7F4A"
C_WEIGHT = "#EAD9B5"
C_WEIGHT_EDGE = "#9A7F4A"
C_DIV = "#222222"
C_ARROW = "#1a1a1a"


def rbox(ax, x, y, w, h, fc, ec, lw=1.35, rs=0.08):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={rs}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ARROW,
            lw=1.2,
            mutation_scale=11,
            shrinkA=0.5,
            shrinkB=0.5,
        ),
        zorder=4,
    )


def label(ax, x, y, s, size=9, weight="bold"):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        ha="center",
        va="center",
        color="#151515",
        zorder=5,
        fontfamily="DejaVu Sans",
    )


def draw_frame(ax, caption: str):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        Rectangle((0.2, 1.2), 9.6, 10.15, fill=False, edgecolor="#111", lw=1.5, zorder=1)
    )
    ax.plot([5.0, 5.0], [1.2, 11.35], color=C_DIV, lw=1.15, zorder=2)
    label(ax, 2.6, 10.95, "GPU 0", size=12)
    label(ax, 7.4, 10.95, "GPU 1", size=12)
    ax.text(
        5.0,
        0.45,
        caption,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111",
        fontfamily="DejaVu Sans",
    )


def draw_weight(ax):
    rbox(ax, 0.5, 1.4, 9.0, 0.8, C_WEIGHT, C_WEIGHT_EDGE, lw=1.3, rs=0.1)
    label(ax, 5.0, 1.8, "Weight Update", size=10.5)


def draw_data(ax, x, y, w, h, text_s: str, dash: str | None = None):
    """dash: None | 'left' | 'right' | 'center' for shard cue like the paper."""
    rbox(ax, x, y, w, h, C_DATA, C_DATA_EDGE, lw=1.3, rs=0.06)
    if dash == "left":
        # dashed left half border cue
        ax.plot([x + w * 0.52, x + w * 0.52], [y + 0.06, y + h - 0.06],
                color=C_DATA_EDGE, lw=1.05, ls=(0, (2.5, 1.8)), zorder=4)
    elif dash == "right":
        ax.plot([x + w * 0.48, x + w * 0.48], [y + 0.06, y + h - 0.06],
                color=C_DATA_EDGE, lw=1.05, ls=(0, (2.5, 1.8)), zorder=4)
    elif dash == "center":
        ax.plot([x + w * 0.5, x + w * 0.5], [y + 0.06, y + h - 0.06],
                color=C_DATA_EDGE, lw=1.05, ls=(0, (2.5, 1.8)), zorder=4)
    label(ax, x + w / 2, y + h / 2, text_s, size=9)


def draw_layer(ax, x, y, w, h, n: int):
    """Full (non-TP) layer box: Tensor / Layer N centered."""
    rbox(ax, x, y, w, h, C_LAYER, C_LAYER_EDGE, lw=1.45, rs=0.1)
    label(ax, x + w / 2, y + h * 0.62, "Tensor", size=8.8)
    label(ax, x + w / 2, y + h * 0.32, f"Layer {n}", size=8.8)


def draw_tp_layer_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    n: int,
    *,
    solid: str = "top",
    size: float = 8.2,
) -> None:
    """
    Per-GPU TP layer box (matches the paper figure):
      - one half solid-filled with full name "Tensor" / "Layer N"
      - other half dashed outline, nearly empty / heavily faded
    solid: "top" (GPU 0 style) or "bottom" (GPU 1 style)
    """
    # Outer rounded frame
    rbox(ax, x, y, w, h, "white", C_LAYER_EDGE, lw=1.45, rs=0.1)
    mid_y = y + h * 0.5
    pad = 0.07

    if solid == "top":
        # solid top half
        rbox(
            ax,
            x + pad,
            mid_y,
            w - 2 * pad,
            h * 0.5 - pad,
            C_LAYER,
            C_LAYER_EDGE,
            lw=1.2,
            rs=0.06,
        )
        # faded / empty bottom half (dashed)
        empty = FancyBboxPatch(
            (x + pad, y + pad),
            w - 2 * pad,
            h * 0.5 - 1.5 * pad,
            boxstyle="round,pad=0.008,rounding_size=0.06",
            facecolor="#F7F2FC",
            edgecolor=C_LAYER_EDGE,
            linewidth=1.25,
            linestyle=(0, (3.5, 2.0)),
            zorder=3,
        )
        ax.add_patch(empty)
        # full name only on solid half; empty half stays blank
        label(ax, x + w / 2, mid_y + h * 0.28, "Tensor", size=size)
        label(ax, x + w / 2, mid_y + h * 0.10, f"Layer {n}", size=size)
    else:
        # faded / empty top half (dashed)
        empty = FancyBboxPatch(
            (x + pad, mid_y),
            w - 2 * pad,
            h * 0.5 - pad,
            boxstyle="round,pad=0.008,rounding_size=0.06",
            facecolor="#F7F2FC",
            edgecolor=C_LAYER_EDGE,
            linewidth=1.25,
            linestyle=(0, (3.5, 2.0)),
            zorder=3,
        )
        ax.add_patch(empty)
        # solid bottom half
        rbox(
            ax,
            x + pad,
            y + pad,
            w - 2 * pad,
            h * 0.5 - 1.5 * pad,
            C_LAYER,
            C_LAYER_EDGE,
            lw=1.2,
            rs=0.06,
        )
        # full name only on solid half; empty half stays blank
        label(ax, x + w / 2, y + h * 0.30, "Tensor", size=size)
        label(ax, x + w / 2, y + h * 0.12, f"Layer {n}", size=size)


def draw_comm(ax, x, y, w=1.35, h=0.68):
    rbox(ax, x, y, w, h, C_COMM, C_COMM_EDGE, lw=1.15, rs=0.07)
    label(ax, x + w / 2, y + h / 2, "gpu-comm", size=7.2)


def draw_dp(ax):
    draw_frame(ax, "(a) Data Parallelism")
    draw_weight(ax)

    draw_data(ax, 1.3, 9.55, 2.6, 0.72, "Data 0", dash="left")
    draw_data(ax, 6.1, 9.55, 2.6, 0.72, "Data 1", dash="right")

    draw_layer(ax, 1.2, 7.2, 2.8, 1.45, 1)
    draw_layer(ax, 6.0, 7.2, 2.8, 1.45, 1)
    draw_layer(ax, 1.2, 4.75, 2.8, 1.45, 2)
    draw_layer(ax, 6.0, 4.75, 2.8, 1.45, 2)

    # vertical arrows
    for cx in (2.6, 7.4):
        arrow(ax, cx, 9.55, cx, 8.7)
        arrow(ax, cx, 7.2, cx, 6.25)
        arrow(ax, cx, 4.75, cx, 3.85)

    draw_comm(ax, 1.9, 3.05)
    draw_comm(ax, 6.7, 3.05)
    arrow(ax, 2.6, 3.05, 2.6, 2.25)
    arrow(ax, 7.4, 3.05, 7.4, 2.25)


def draw_tp(ax):
    draw_frame(ax, "(b) Tensor Parallelism")
    draw_weight(ax)

    # Data straddling the divider
    draw_data(ax, 3.25, 9.55, 3.5, 0.72, "Data", dash="center")

    # Per-GPU TP boxes: GPU0 solid-top / empty-bottom; GPU1 empty-top / solid-bottom
    draw_tp_layer_box(ax, 1.2, 7.15, 2.8, 1.55, 1, solid="top")
    draw_tp_layer_box(ax, 6.0, 7.15, 2.8, 1.55, 1, solid="bottom")
    draw_tp_layer_box(ax, 1.2, 4.65, 2.8, 1.55, 2, solid="top")
    draw_tp_layer_box(ax, 6.0, 4.65, 2.8, 1.55, 2, solid="bottom")

    # Data -> Layer 1
    arrow(ax, 4.2, 9.55, 2.6, 8.75)
    arrow(ax, 5.8, 9.55, 7.4, 8.75)

    # Inter-GPU gpu-comm at layer 1
    draw_comm(ax, 4.325, 7.58)
    arrow(ax, 4.0, 7.92, 4.325, 7.92)
    arrow(ax, 5.675, 7.92, 6.0, 7.92)

    # Layer1 -> Layer2
    arrow(ax, 2.6, 7.15, 2.6, 6.25)
    arrow(ax, 7.4, 7.15, 7.4, 6.25)

    # Inter-GPU gpu-comm at layer 2
    draw_comm(ax, 4.325, 5.08)
    arrow(ax, 4.0, 5.42, 4.325, 5.42)
    arrow(ax, 5.675, 5.42, 6.0, 5.42)

    # Merge gpu-comm -> weight update
    draw_comm(ax, 4.325, 3.05)
    arrow(ax, 2.6, 4.65, 4.7, 3.75)
    arrow(ax, 7.4, 4.65, 5.3, 3.75)
    arrow(ax, 5.0, 3.05, 5.0, 2.25)


def draw_pp(ax):
    draw_frame(ax, "(c) Pipeline Parallelism")
    draw_weight(ax)

    draw_data(ax, 1.3, 9.55, 2.6, 0.72, "Data")

    # Layer 1 on GPU0, Layer 2 on GPU1 (same height)
    draw_layer(ax, 1.2, 6.35, 2.8, 1.7, 1)
    draw_layer(ax, 6.0, 6.35, 2.8, 1.7, 2)

    arrow(ax, 2.6, 9.55, 2.6, 8.1)

    draw_comm(ax, 4.325, 6.85)
    arrow(ax, 4.0, 7.19, 4.325, 7.19)
    arrow(ax, 5.675, 7.19, 6.0, 7.19)

    # Layer2 down to weight update
    arrow(ax, 7.4, 6.35, 7.4, 2.25)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.4))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.05, wspace=0.06)

    draw_dp(axes[0])
    draw_tp(axes[1])
    draw_pp(axes[2])

    for ext in ("pdf", "png"):
        path = OUT / f"parallelism_dp_tp_pp.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
