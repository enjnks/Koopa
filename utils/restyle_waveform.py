"""Restyle a raster waveform chart to the shared publication figure format."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TARGET_BLUE = "#1F77B4"
FIGURE_SIZE_INCHES = (4.10, 2.96)
OUTPUT_DPI = 100


def extract_waveform_overlay(
    source_path: Path,
    plot_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract the blue signal as a transparent, recolored RGBA overlay."""
    source = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.float32)
    left, top, right, bottom = plot_bounds
    plot = source[top:bottom, left:right]
    red, green, blue = plot[..., 0], plot[..., 1], plot[..., 2]

    # The source signal is pure blue on white. Channel distance gives the
    # antialias coverage without retaining the old RGB value.
    coverage = np.clip((blue - np.maximum(red, green)) / 255.0, 0.0, 1.0)
    overlay = np.zeros((*coverage.shape, 4), dtype=np.float32)
    overlay[..., 0] = 31 / 255
    overlay[..., 1] = 119 / 255
    overlay[..., 2] = 180 / 255
    overlay[..., 3] = coverage
    return overlay


def render_chart(
    source_path: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Render the waveform with matched dimensions, typography, and colors."""
    overlay = extract_waveform_overlay(
        source_path,
        # Interior of the original axes, excluding its old spines and labels.
        plot_bounds=(198, 31, 1440, 759),
    )

    plt.rcParams.update(
        {
            # Tinos is metrically compatible with Times New Roman.
            "font.family": "Tinos",
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.0,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "text.color": "#000000",
            "svg.fonttype": "none",
        }
    )

    figure, axes = plt.subplots(
        figsize=FIGURE_SIZE_INCHES,
        dpi=OUTPUT_DPI,
        facecolor="#FFFFFF",
    )
    axes.set_facecolor("#FFFFFF")
    axes.imshow(
        overlay,
        extent=(0, 12200, -22.5, 27.5),
        origin="upper",
        interpolation="lanczos",
        aspect="auto",
        zorder=2,
    )

    axes.set_xlim(0, 12200)
    axes.set_ylim(-22.5, 27.5)
    axes.set_xticks(np.arange(0, 12001, 2000))
    axes.set_yticks([-20, -10, 0, 10, 20])
    axes.set_xlabel("Time(min)", fontsize=16, labelpad=4)
    axes.set_ylabel("Amplitude(g)", fontsize=16, labelpad=5)
    axes.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        direction="out",
        length=4,
        width=1,
        pad=4,
    )
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    # Match the approximately 410×296 reference layout.
    # Leave extra room for the five-digit "12000" tick label.
    figure.subplots_adjust(left=0.172, right=0.945, bottom=0.205, top=0.975)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        facecolor="#FFFFFF",
        transparent=False,
    )
    figure.savefig(
        svg_path,
        facecolor="#FFFFFF",
        transparent=False,
    )
    plt.close(figure)

    # Matplotlib path output may contain line-ending spaces; normalize them so
    # repository whitespace checks remain clean.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("png_output", type=Path)
    parser.add_argument("svg_output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_chart(args.source, args.png_output, args.svg_output)
    print(f"saved PNG: {args.png_output}")
    print(f"saved SVG: {args.svg_output}")


if __name__ == "__main__":
    main()
