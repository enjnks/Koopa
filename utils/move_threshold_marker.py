"""Move one dashed threshold marker while preserving the source chart.

The utility is intentionally pixel-based: it keeps every source pixel outside
the old and new marker strips unchanged, including transparency, axes, labels,
legend, and the feature curve.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from PIL import Image


def is_orange(pixel: tuple[int, int, int, int]) -> bool:
    """Return whether an RGBA pixel belongs to the orange marker core."""
    red, green, blue, alpha = pixel
    return (
        alpha > 20
        and red > 80
        and red > green * 1.12
        and green > blue * 1.8
    )


def is_blue(pixel: tuple[int, int, int, int]) -> bool:
    """Protect feature-curve pixels when collecting marker antialiasing."""
    red, green, blue, alpha = pixel
    return alpha > 10 and blue > red * 1.25 and blue > green * 1.25


def interpolate_rgba(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    """Linearly interpolate two straight-alpha RGBA pixels."""
    return tuple(
        round(left[channel] * (1 - ratio) + right[channel] * ratio)
        for channel in range(4)
    )


def move_marker(
    source_path: Path,
    png_path: Path,
    svg_path: Path,
    *,
    old_center_x: int,
    new_center_x: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Move the orange vertical marker and return old/new changed bounds."""
    source = Image.open(source_path).convert("RGBA")
    result = source.copy()
    pixels = source.load()
    output = result.load()

    # Detect the orange core only around the old vertical marker, excluding the
    # legend sample. The marker spans the plot vertically.
    core = {
        (x, y)
        for y in range(source.height)
        for x in range(max(0, old_center_x - 8), min(source.width, old_center_x + 9))
        if is_orange(pixels[x, y])
    }
    if not core:
        raise ValueError("No orange marker pixels found at the supplied position")

    # Include antialiased/shadow pixels surrounding the orange core, but never
    # absorb the blue feature curve into the movable marker.
    marker_mask: set[tuple[int, int]] = set()
    for core_x, core_y in core:
        for y in range(max(0, core_y - 2), min(source.height, core_y + 3)):
            for x in range(max(0, core_x - 4), min(source.width, core_x + 5)):
                pixel = pixels[x, y]
                if pixel[3] > 0 and not is_blue(pixel):
                    marker_mask.add((x, y))

    min_x = min(x for x, _ in marker_mask)
    max_x = max(x for x, _ in marker_mask)
    min_y = min(y for _, y in marker_mask)
    max_y = max(y for _, y in marker_mask)
    left_x = max(0, min_x - 3)
    right_x = min(source.width - 1, max_x + 3)

    # Restore what the old line covered from the nearest unaffected pixels.
    # This is exact for transparent background and horizontal axes; around the
    # blue curve it creates a short, continuous interpolation across the strip.
    for x, y in marker_mask:
        ratio = (x - left_x) / (right_x - left_x)
        output[x, y] = interpolate_rgba(
            pixels[left_x, y],
            pixels[right_x, y],
            ratio,
        )

    # Alpha-composite the original marker pixels at their calibrated location.
    delta_x = new_center_x - old_center_x
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    for x, y in marker_mask:
        target_x = x + delta_x
        if 0 <= target_x < source.width:
            overlay_pixels[target_x, y] = pixels[x, y]
    result = Image.alpha_composite(result, overlay)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(png_path, format="PNG", optimize=False)

    encoded_png = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg_path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{source.width}" height="{source.height}" '
            f'viewBox="0 0 {source.width} {source.height}">\n'
            f'  <image width="{source.width}" height="{source.height}" '
            f'href="data:image/png;base64,{encoded_png}"/>\n'
            f"</svg>\n"
        ),
        encoding="utf-8",
    )

    old_bounds = (min_x, min_y, max_x, max_y)
    new_bounds = (
        min_x + delta_x,
        min_y,
        max_x + delta_x,
        max_y,
    )
    return old_bounds, new_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("png_output", type=Path)
    parser.add_argument("svg_output", type=Path)
    parser.add_argument("--old-x", type=int, required=True)
    parser.add_argument("--new-x", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_bounds, new_bounds = move_marker(
        args.source,
        args.png_output,
        args.svg_output,
        old_center_x=args.old_x,
        new_center_x=args.new_x,
    )
    print(f"old marker bounds: {old_bounds}")
    print(f"new marker bounds: {new_bounds}")


if __name__ == "__main__":
    main()
