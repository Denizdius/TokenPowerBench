#!/usr/bin/env python3
"""Paper-style hybrid parallelism schematics: TP×PP, TP×DP, DP×PP (forward only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"

# Match generate_parallelism_figure.py
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
C_GROUP = "#888888"


def rbox(ax, x, y, w, h, fc, ec, lw=1.3, rs=0.07):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={rs}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, lw=1.15, ms=10):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ARROW,
            lw=lw,
            mutation_scale=ms,
            shrinkA=0.4,
            shrinkB=0.4,
        ),
        zorder=4,
    )


def label(ax, x, y, s, size=8.5, weight="bold"):
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


def draw_frame_4gpu(ax, caption: str, group_labels: tuple[str, str] | None = None):
    """Four GPU columns. Optional group labels under pairs (e.g. Replica / Stage)."""
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 12.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        Rectangle((0.2, 1.15), 15.8, 10.55, fill=False, edgecolor="#111", lw=1.5, zorder=1)
    )

    # Column dividers at 4.15, 8.1, 12.05
    for x in (4.15, 8.1, 12.05):
        ax.plot([x, x], [1.15, 11.7], color=C_DIV, lw=1.05, zorder=2)

    # Mid divider slightly thicker (separates the two hybrid groups)
    ax.plot([8.1, 8.1], [1.15, 11.7], color=C_DIV, lw=1.55, zorder=2)

    centers = [2.175, 6.125, 10.075, 14.025]
    for i, cx in enumerate(centers):
        label(ax, cx, 11.35, f"GPU {i}", size=10)

    if group_labels:
        # Above GPU headers: group tags for left pair / right pair
        label(ax, 4.15, 10.85, group_labels[0], size=8.2)
        label(ax, 12.05, 10.85, group_labels[1], size=8.2)

    ax.text(
        8.1,
        0.4,
        caption,
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111",
        fontfamily="DejaVu Sans",
    )


def draw_weight(ax):
    rbox(ax, 0.45, 1.35, 15.3, 0.75, C_WEIGHT, C_WEIGHT_EDGE, lw=1.25, rs=0.09)
    label(ax, 8.1, 1.72, "Output Tokens", size=10)


def draw_data(ax, x, y, w, h, text_s: str, dash: str | None = None):
    rbox(ax, x, y, w, h, C_DATA, C_DATA_EDGE, lw=1.2, rs=0.05)
    if dash in ("left", "right", "center"):
        frac = 0.52 if dash == "left" else (0.48 if dash == "right" else 0.5)
        ax.plot(
            [x + w * frac, x + w * frac],
            [y + 0.05, y + h - 0.05],
            color=C_DATA_EDGE,
            lw=1.0,
            ls=(0, (2.5, 1.8)),
            zorder=4,
        )
    label(ax, x + w / 2, y + h / 2, text_s, size=8.2)


def draw_layer(ax, x, y, w, h, n: int):
    """Full (non-TP) layer box."""
    rbox(ax, x, y, w, h, C_LAYER, C_LAYER_EDGE, lw=1.35, rs=0.08)
    label(ax, x + w / 2, y + h * 0.62, "Tensor", size=7.6)
    label(ax, x + w / 2, y + h * 0.30, f"Layer {n}", size=7.6)


def draw_tp_layer_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    n: int,
    *,
    solid: str = "top",
    size: float = 6.8,
) -> None:
    """
    Per-GPU TP layer box (paper style):
      solid half filled with full "Tensor" / "Layer N"
      other half dashed, nearly empty / faded, also labeled fully
    solid: "top" or "bottom"
    """
    from matplotlib.patches import FancyBboxPatch

    rbox(ax, x, y, w, h, "white", C_LAYER_EDGE, lw=1.3, rs=0.08)
    mid_y = y + h * 0.5
    pad = 0.06

    if solid == "top":
        rbox(
            ax, x + pad, mid_y, w - 2 * pad, h * 0.5 - pad,
            C_LAYER, C_LAYER_EDGE, lw=1.1, rs=0.05,
        )
        empty = FancyBboxPatch(
            (x + pad, y + pad),
            w - 2 * pad,
            h * 0.5 - 1.5 * pad,
            boxstyle="round,pad=0.006,rounding_size=0.05",
            facecolor="#F7F2FC",
            edgecolor=C_LAYER_EDGE,
            linewidth=1.15,
            linestyle=(0, (3.2, 1.8)),
            zorder=3,
        )
        ax.add_patch(empty)
        # full name only on solid half; empty half stays blank
        label(ax, x + w / 2, mid_y + h * 0.28, "Tensor", size=size)
        label(ax, x + w / 2, mid_y + h * 0.10, f"Layer {n}", size=size)
    else:
        empty = FancyBboxPatch(
            (x + pad, mid_y),
            w - 2 * pad,
            h * 0.5 - pad,
            boxstyle="round,pad=0.006,rounding_size=0.05",
            facecolor="#F7F2FC",
            edgecolor=C_LAYER_EDGE,
            linewidth=1.15,
            linestyle=(0, (3.2, 1.8)),
            zorder=3,
        )
        ax.add_patch(empty)
        rbox(
            ax, x + pad, y + pad, w - 2 * pad, h * 0.5 - 1.5 * pad,
            C_LAYER, C_LAYER_EDGE, lw=1.1, rs=0.05,
        )
        # full name only on solid half; empty half stays blank
        label(ax, x + w / 2, y + h * 0.30, "Tensor", size=size)
        label(ax, x + w / 2, y + h * 0.12, f"Layer {n}", size=size)


def draw_comm(ax, x, y, w=1.2, h=0.58, size=7.0):
    rbox(ax, x, y, w, h, C_COMM, C_COMM_EDGE, lw=1.05, rs=0.06)
    label(ax, x + w / 2, y + h / 2, "comm", size=size)


def draw_tp_pp(ax):
    """
    TP2 × PP2:
      Stage 0 (PP): GPU0–GPU1 hold Tensor Layer 1 with TP
      Stage 1 (PP): GPU2–GPU3 hold Tensor Layer 2 with TP
    """
    draw_frame_4gpu(ax, "(a) TP × PP", ("PP stage 0", "PP stage 1"))
    draw_weight(ax)

    # Shared data for stage 0 (TP shard of same microbatch)
    # Ideal batch box size (same across all hybrid panels)
    BW, BH = 2.3, 0.62

    draw_data(ax, 3.0, 9.55, BW, BH, "Batch")

    # Stage 0: Layer 1 TP on GPU0|GPU1
    draw_tp_layer_box(ax, 1.15, 7.05, 2.05, 1.45, 1, solid="top", size=6.4)
    draw_tp_layer_box(ax, 5.1, 7.05, 2.05, 1.45, 1, solid="bottom", size=6.4)
    # Stage 1: Layer 2 TP on GPU2|GPU3
    draw_tp_layer_box(ax, 9.05, 7.05, 2.05, 1.45, 2, solid="top", size=6.4)
    draw_tp_layer_box(ax, 13.0, 7.05, 2.05, 1.45, 2, solid="bottom", size=6.4)

    # Data -> stage0
    arrow(ax, 3.7, 9.55, 2.175, 8.55)
    arrow(ax, 4.6, 9.55, 6.125, 8.55)

    # TP comm on stage-0
    draw_comm(ax, 3.55, 7.45, w=1.05, h=0.55, size=6.5)
    arrow(ax, 3.2, 7.72, 3.55, 7.72)
    arrow(ax, 4.6, 7.72, 5.1, 7.72)

    # PP handoff: Stage 0 (GPU1 Layer 1) -> Stage 1 (GPU2 Layer 2 only)
    draw_comm(ax, 7.5, 5.25, w=1.2, h=0.55, size=6.5)
    arrow(ax, 6.125, 7.05, 8.1, 5.85)
    arrow(ax, 8.1, 5.25, 10.075, 7.05)

    # TP comm between GPU2 and GPU3 Tensor Layer 2
    draw_comm(ax, 11.45, 7.45, w=1.05, h=0.55, size=6.5)
    arrow(ax, 11.1, 7.72, 11.45, 7.72)
    arrow(ax, 12.5, 7.72, 13.0, 7.72)

    # Only GPU3 Tensor Layer 2 -> Weight Update
    arrow(ax, 14.025, 7.05, 14.025, 2.15)


def draw_tp_dp(ax):
    """
    TP2 × DP2:
      Replica 0: GPU0–GPU1, Data 0, full model with TP
      Replica 1: GPU2–GPU3, Data 1, full model with TP
    """
    draw_frame_4gpu(ax, "(b) TP × DP", ("DP replica 0", "DP replica 1"))
    draw_weight(ax)

    # Same batch box size as TP×PP; centered over each DP replica pair
    BW, BH = 2.3, 0.62
    draw_data(ax, 3.0, 9.55, BW, BH, "Batch 0")   # over GPU0|GPU1
    draw_data(ax, 10.9, 9.55, BW, BH, "Batch 1")  # over GPU2|GPU3

    # Replica 0: TP on GPU0|GPU1
    draw_tp_layer_box(ax, 1.15, 7.25, 2.05, 1.25, 1, solid="top", size=6.2)
    draw_tp_layer_box(ax, 5.1, 7.25, 2.05, 1.25, 1, solid="bottom", size=6.2)
    draw_tp_layer_box(ax, 1.15, 5.05, 2.05, 1.25, 2, solid="top", size=6.2)
    draw_tp_layer_box(ax, 5.1, 5.05, 2.05, 1.25, 2, solid="bottom", size=6.2)
    # Replica 1: TP on GPU2|GPU3
    draw_tp_layer_box(ax, 9.05, 7.25, 2.05, 1.25, 1, solid="top", size=6.2)
    draw_tp_layer_box(ax, 13.0, 7.25, 2.05, 1.25, 1, solid="bottom", size=6.2)
    draw_tp_layer_box(ax, 9.05, 5.05, 2.05, 1.25, 2, solid="top", size=6.2)
    draw_tp_layer_box(ax, 13.0, 5.05, 2.05, 1.25, 2, solid="bottom", size=6.2)

    # Batch -> L1 (fan out from each batch box to that replica's TP ranks)
    arrow(ax, 3.7, 9.55, 2.175, 8.55)
    arrow(ax, 4.6, 9.55, 6.125, 8.55)
    arrow(ax, 11.6, 9.55, 10.075, 8.55)
    arrow(ax, 12.5, 9.55, 14.025, 8.55)

    # TP comm L1
    draw_comm(ax, 3.55, 7.55, w=1.05, h=0.5, size=6.5)
    arrow(ax, 3.2, 7.8, 3.55, 7.8)
    arrow(ax, 4.6, 7.8, 5.1, 7.8)
    draw_comm(ax, 11.45, 7.55, w=1.05, h=0.5, size=6.5)
    arrow(ax, 11.1, 7.8, 11.45, 7.8)
    arrow(ax, 12.5, 7.8, 13.0, 7.8)

    # L1 -> L2
    arrow(ax, 2.175, 7.25, 2.175, 6.35)
    arrow(ax, 6.125, 7.25, 6.125, 6.35)
    arrow(ax, 10.075, 7.25, 10.075, 6.35)
    arrow(ax, 14.025, 7.25, 14.025, 6.35)

    # TP comm L2
    draw_comm(ax, 3.55, 5.35, w=1.05, h=0.5, size=6.5)
    arrow(ax, 3.2, 5.6, 3.55, 5.6)
    arrow(ax, 4.6, 5.6, 5.1, 5.6)
    draw_comm(ax, 11.45, 5.35, w=1.05, h=0.5, size=6.5)
    arrow(ax, 11.1, 5.6, 11.45, 5.6)
    arrow(ax, 12.5, 5.6, 13.0, 5.6)

    # DP comm then weight update
    draw_comm(ax, 3.55, 3.35, w=1.05, h=0.5, size=6.5)
    draw_comm(ax, 11.45, 3.35, w=1.05, h=0.5, size=6.5)
    draw_comm(ax, 7.5, 3.35, w=1.2, h=0.5, size=6.5)
    arrow(ax, 2.175, 5.05, 4.0, 3.9)
    arrow(ax, 6.125, 5.05, 4.2, 3.9)
    arrow(ax, 10.075, 5.05, 12.0, 3.9)
    arrow(ax, 14.025, 5.05, 12.2, 3.9)
    arrow(ax, 4.6, 3.6, 7.5, 3.6)
    arrow(ax, 11.45, 3.6, 8.7, 3.6)
    arrow(ax, 8.1, 3.35, 8.1, 2.15)


def draw_dp_pp(ax):
    """
    DP2 × PP2:
      Replica 0: GPU0 (Layer 1) --PP--> GPU1 (Layer 2), Data 0
      Replica 1: GPU2 (Layer 1) --PP--> GPU3 (Layer 2), Data 1
    """
    draw_frame_4gpu(ax, "(c) DP × PP", ("DP replica 0", "DP replica 1"))
    draw_weight(ax)

    # Same batch box size as TP×PP; centered over each replica's entry GPU
    BW, BH = 2.3, 0.62
    draw_data(ax, 1.025, 9.55, BW, BH, "Batch 0")  # over GPU0
    draw_data(ax, 8.925, 9.55, BW, BH, "Batch 1")  # over GPU2

    # Replica 0: L1 on GPU0, L2 on GPU1 (no TP — full boxes)
    draw_layer(ax, 1.15, 6.7, 2.05, 1.55, 1)
    draw_layer(ax, 5.1, 6.7, 2.05, 1.55, 2)

    # Replica 1: L1 on GPU2, L2 on GPU3
    draw_layer(ax, 9.05, 6.7, 2.05, 1.55, 1)
    draw_layer(ax, 13.0, 6.7, 2.05, 1.55, 2)

    # Data -> L1
    arrow(ax, 2.175, 9.55, 2.175, 8.3)
    arrow(ax, 10.075, 9.55, 10.075, 8.3)

    # PP comm within each replica
    draw_comm(ax, 3.55, 7.2, w=1.05, h=0.55, size=6.5)
    arrow(ax, 3.2, 7.48, 3.55, 7.48)
    arrow(ax, 4.6, 7.48, 5.1, 7.48)

    draw_comm(ax, 11.45, 7.2, w=1.05, h=0.55, size=6.5)
    arrow(ax, 11.1, 7.48, 11.45, 7.48)
    arrow(ax, 12.5, 7.48, 13.0, 7.48)

    # DP comm across replicas before weight update
    draw_comm(ax, 3.55, 3.55, w=1.05, h=0.5, size=6.5)
    draw_comm(ax, 11.45, 3.55, w=1.05, h=0.5, size=6.5)
    draw_comm(ax, 7.5, 3.55, w=1.2, h=0.5, size=6.5)

    arrow(ax, 6.125, 6.7, 4.1, 4.1)
    arrow(ax, 14.025, 6.7, 12.0, 4.1)
    arrow(ax, 4.6, 3.8, 7.5, 3.8)
    arrow(ax, 11.45, 3.8, 8.7, 3.8)
    arrow(ax, 8.1, 3.55, 8.1, 2.15)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.05, wspace=0.04)

    draw_tp_pp(axes[0])
    draw_tp_dp(axes[1])
    draw_dp_pp(axes[2])

    for ext in ("pdf", "png"):
        path = OUT / f"parallelism_hybrid_tpxpp_tpxdp_dpxpp.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
