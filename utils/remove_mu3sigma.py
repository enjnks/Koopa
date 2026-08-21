"""Remove the mu+3sigma marker and legend row from a transparent PNG chart."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from PIL import Image


def is_orange(pixel: tuple[int, int, int, int]) -> bool:
    """Identify orange/yellow line pixels, including partially transparent ones."""
    red, green, blue, alpha = pixel
    return (
        alpha > 10
        and red > 60
        and red > green * 1.1
        and green > blue * 1.8
    )


def is_blue(pixel: tuple[int, int, int, int]) -> bool:
    """Protect feature-curve pixels where they cross the orange marker."""
    red, green, blue, alpha = pixel
    return alpha > 10 and blue > 25 and blue > red * 1.25 and blue > green * 1.25


def remove_vertical_marker(
    image: Image.Image,
    *,
    center_x: int,
    half_width: int = 4,
) -> tuple[int, int, int, int]:
    """Erase the orange marker footprint while retaining overlapping blue pixels."""
    source = image.copy()
    source_pixels = source.load()
    output = image.load()

    core_rows = {
        y
        for y in range(image.height)
        for x in range(max(0, center_x - half_width), min(image.width, center_x + half_width + 1))
        if is_orange(source_pixels[x, y])
    }
    if not core_rows:
        raise ValueError("No orange vertical marker found")

    rows = {
        adjacent_y
        for y in core_rows
        for adjacent_y in range(max(0, y - 1), min(image.height, y + 2))
    }
    left = max(0, center_x - half_width)
    right = min(image.width - 1, center_x + half_width)

    for y in rows:
        for x in range(left, right + 1):
            pixel = source_pixels[x, y]
            if not is_blue(pixel):
                output[x, y] = (0, 0, 0, 0)

    # Restore the horizontal top and bottom axes where the marker crossed them.
    for y in rows:
        left_sample = source_pixels[max(0, left - 3), y]
        right_sample = source_pixels[min(image.width - 1, right + 3), y]
        samples_are_axis = (
            left_sample[3] > 0
            and right_sample[3] > 0
            and max(left_sample[:3]) < 80
            and max(right_sample[:3]) < 80
        )
        if samples_are_axis:
            for x in range(left, right + 1):
                ratio = (x - left) / max(1, right - left)
                output[x, y] = tuple(
                    round(left_sample[channel] * (1 - ratio) + right_sample[channel] * ratio)
                    for channel in range(4)
                )

    return left, min(rows), right, max(rows)


def compact_legend(
    image: Image.Image,
    *,
    bounds: tuple[int, int, int, int],
    removed_row: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Delete the middle legend row and move the final row upward unchanged."""
    left, top, right, bottom = bounds
    remove_top, remove_bottom = removed_row
    source = image.copy()

    # The plot behind this legend area is transparent, so clearing the old
    # rectangle restores the original plot background exactly.
    transparent = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
    image.paste(transparent, (left, top))

    upper = source.crop((left, top, right, remove_top))
    lower = source.crop((left, remove_bottom, right, bottom))
    compact = Image.new(
        "RGBA",
        (right - left, upper.height + lower.height),
        (0, 0, 0, 0),
    )
    compact.alpha_composite(upper, (0, 0))
    compact.alpha_composite(lower, (0, upper.height))
    image.alpha_composite(compact, (left, top))

    return left, top, right, top + compact.height


def save_png_and_svg(image: Image.Image, png_path: Path, svg_path: Path) -> None:
    """Save the edited RGBA image and a self-contained SVG wrapper."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG", optimize=False)
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg_path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{image.width}" height="{image.height}" '
            f'viewBox="0 0 {image.width} {image.height}">\n'
            f'  <image width="{image.width}" height="{image.height}" '
            f'href="data:image/png;base64,{encoded}"/>\n'
            "</svg>\n"
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("png_output", type=Path)
    parser.add_argument("svg_output", type=Path)
    parser.add_argument("--marker-x", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.source).convert("RGBA")
    marker_bounds = remove_vertical_marker(image, center_x=args.marker_x)
    legend_bounds = compact_legend(
        image,
        bounds=(82, 18, 193, 95),
        removed_row=(43, 66),
    )
    save_png_and_svg(image, args.png_output, args.svg_output)
    print(f"removed marker bounds: {marker_bounds}")
    print(f"compacted legend bounds: {legend_bounds}")


if __name__ == "__main__":
    main()
