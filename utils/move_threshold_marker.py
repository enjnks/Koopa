"""Move one dashed threshold marker while preserving the source chart.

The utility is intentionally pixel-based: it keeps every source pixel outside
the old and new marker strips unchanged, including transparency, axes, labels,
legend, and the feature curve.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from PIL import Image, ImageDraw


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


def unpremultiply_region(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
) -> None:
    """Correct premultiplied colors stored in a straight-alpha PNG region.

    Some WPS exports store a semitransparent white legend as (203, 203, 203,
    203), which Word composites as gray. Straight-alpha PNG requires the RGB
    channels to remain (255, 255, 255) at that alpha.
    """
    pixels = image.load()
    left, top, right, bottom = bounds
    for y in range(max(0, top), min(image.height, bottom)):
        for x in range(max(0, left), min(image.width, right)):
            red, green, blue, alpha = pixels[x, y]
            if 0 < alpha < 255:
                corrected = (
                    min(255, round(red * 255 / alpha)),
                    min(255, round(green * 255 / alpha)),
                    min(255, round(blue * 255 / alpha)),
                )
                # Remove tiny channel-rounding differences from the white fill
                # without touching dark text, borders, or colored line samples.
                if min(corrected) >= 245 and max(corrected) - min(corrected) <= 10:
                    corrected = (255, 255, 255)
                pixels[x, y] = (*corrected, alpha)


def normalize_line_colors(image: Image.Image) -> dict[str, int]:
    """Set chart-line RGB channels to their original Matplotlib colors.

    Alpha values and pixel coordinates are left untouched, so antialiasing,
    line thickness, dash patterns, and marker positions remain unchanged.
    """
    pixels = image.load()
    changed = {"blue": 0, "orange": 0, "green": 0}
    for y in range(image.height):
        for x in range(image.width):
            red, green, blue, alpha = pixels[x, y]
            if alpha == 0:
                continue
            in_legend = (
                image.width * 0.19 <= x <= image.width * 0.45
                and image.height * 0.05 <= y <= image.height * 0.30
            )
            in_plot = (
                image.width * 0.21 <= x <= image.width * 0.945
                and image.height * 0.04 <= y <= image.height * 0.775
                and not in_legend
            )
            in_blue_legend = (
                image.width * 0.20 <= x <= image.width * 0.30
                and image.height * 0.08 <= y <= image.height * 0.115
            )
            in_orange_legend = (
                image.width * 0.20 <= x <= image.width * 0.30
                and image.height * 0.145 <= y <= image.height * 0.185
            )
            in_green_legend = (
                image.width * 0.20 <= x <= image.width * 0.30
                and image.height * 0.215 <= y <= image.height * 0.255
            )
            in_green_marker = (
                image.width * 0.735 <= x <= image.width * 0.765
                and image.height * 0.03 <= y <= image.height * 0.785
            )
            in_orange_marker = (
                image.width * 0.76 <= x <= image.width * 0.79
                and image.height * 0.03 <= y <= image.height * 0.785
            )

            if (
                (in_plot or in_blue_legend)
                and blue > red * 1.25
                and blue > green * 1.25
            ):
                pixels[x, y] = (0, 0, 255, alpha)
                changed["blue"] += 1
            elif (
                (in_green_marker or in_green_legend)
                and green > red * 1.25
                and green > blue * 1.25
            ):
                pixels[x, y] = (0, 128, 0, alpha)
                changed["green"] += 1
            elif (
                (in_orange_marker or in_orange_legend)
                and red > green * 1.1
                and green > blue * 1.8
            ):
                pixels[x, y] = (255, 165, 0, alpha)
                changed["orange"] += 1
    return changed


def move_marker(
    source_path: Path,
    png_path: Path,
    svg_path: Path,
    *,
    old_center_x: int,
    new_center_x: int,
    restore_blue_path: list[tuple[int, int]] | None = None,
    normalize_legend_bounds: tuple[int, int, int, int] | None = None,
    normalize_colors: bool = False,
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
    restore_mask: set[tuple[int, int]] = set()
    for core_x, core_y in core:
        for y in range(max(0, core_y - 2), min(source.height, core_y + 3)):
            for x in range(max(0, core_x - 4), min(source.width, core_x + 5)):
                restore_mask.add((x, y))
                pixel = pixels[x, y]
                if pixel[3] > 0 and not is_blue(pixel):
                    marker_mask.add((x, y))

    min_x = min(x for x, _ in marker_mask)
    max_x = max(x for x, _ in marker_mask)
    min_y = min(y for _, y in marker_mask)
    max_y = max(y for _, y in marker_mask)
    # WPS adds a faint antialiasing shadow beyond the colored marker core.
    # Clear the complete narrow strip so no orange/gray residue remains.
    restore_min_x = max(0, old_center_x - 9)
    restore_max_x = min(source.width - 1, old_center_x + 9)
    restore_mask = {
        (x, y)
        for y in range(min_y, max_y + 1)
        for x in range(restore_min_x, restore_max_x + 1)
    }
    left_x = max(0, restore_min_x - 3)
    right_x = min(source.width - 1, restore_max_x + 3)

    # Remove the complete old marker footprint. Transparent plot pixels become
    # transparent again, while the horizontal plot borders are sampled from
    # their unchanged neighboring pixels.
    for x, y in restore_mask:
        if y <= min_y + 5 or y >= max_y - 5:
            ratio = (x - left_x) / (right_x - left_x)
            output[x, y] = interpolate_rgba(
                pixels[left_x, y],
                pixels[right_x, y],
                ratio,
            )
        else:
            output[x, y] = (0, 0, 0, 0)

    # The old marker hid a short part of the feature curve. If supplied,
    # reconnect that path with an antialiased stroke sampled from the source.
    if restore_blue_path:
        scale = 4
        blue_overlay = Image.new(
            "RGBA",
            (source.width * scale, source.height * scale),
            (0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(blue_overlay)
        draw.line(
            [(x * scale, y * scale) for x, y in restore_blue_path],
            fill=(10, 10, 188, 255),
            width=3 * scale,
            joint="curve",
        )
        blue_overlay = blue_overlay.resize(source.size, Image.Resampling.LANCZOS)
        result = Image.alpha_composite(result, blue_overlay)

    # Alpha-composite the original marker pixels at their calibrated location.
    delta_x = new_center_x - old_center_x
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    for x, y in marker_mask:
        target_x = x + delta_x
        if 0 <= target_x < source.width:
            overlay_pixels[target_x, y] = pixels[x, y]
    result = Image.alpha_composite(result, overlay)

    if normalize_legend_bounds:
        unpremultiply_region(result, normalize_legend_bounds)
    if normalize_colors:
        normalize_line_colors(result)

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

    old_bounds = (restore_min_x, min_y, restore_max_x, max_y)
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
    parser.add_argument(
        "--restore-blue-path",
        help='Semicolon-separated points, for example "1261,190;1266,187;1266,110"',
    )
    parser.add_argument(
        "--normalize-legend",
        help='Legend bounds as "left,top,right,bottom"',
    )
    parser.add_argument(
        "--normalize-colors",
        action="store_true",
        help="Use blue #0000FF, orange #FFA500, and green #008000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restore_blue_path = None
    if args.restore_blue_path:
        restore_blue_path = [
            tuple(map(int, point.split(",",)))
            for point in args.restore_blue_path.split(";")
        ]
    normalize_legend_bounds = None
    if args.normalize_legend:
        normalize_legend_bounds = tuple(map(int, args.normalize_legend.split(",")))

    old_bounds, new_bounds = move_marker(
        args.source,
        args.png_output,
        args.svg_output,
        old_center_x=args.old_x,
        new_center_x=args.new_x,
        restore_blue_path=restore_blue_path,
        normalize_legend_bounds=normalize_legend_bounds,
        normalize_colors=args.normalize_colors,
    )
    print(f"old marker bounds: {old_bounds}")
    print(f"new marker bounds: {new_bounds}")


if __name__ == "__main__":
    main()
