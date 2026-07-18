#!/usr/bin/env python3
"""Generate the quiet monochrome waveform-to-caret app icon."""

import sys

from AppKit import NSBitmapImageRep, NSPNGFileType
from Quartz import (
    CGBitmapContextCreate,
    CGBitmapContextCreateImage,
    CGColorSpaceCreateDeviceRGB,
    CGContextAddArc,
    CGContextAddLineToPoint,
    CGContextBeginPath,
    CGContextFillPath,
    CGContextMoveToPoint,
    CGContextSetRGBFillColor,
    kCGImageAlphaPremultipliedLast,
)

BACKGROUND_TONE = (0.15, 0.15, 0.15, 1.0)
MARK_TONE = (0.88, 0.88, 0.88, 1.0)


def generate_icon(size, output_path):
    """Draw a waveform flowing into a text insertion caret."""
    color_space = CGColorSpaceCreateDeviceRGB()
    context = CGBitmapContextCreate(
        None,
        size,
        size,
        8,
        size * 4,
        color_space,
        kCGImageAlphaPremultipliedLast,
    )

    scale = float(size)
    _set_fill(context, BACKGROUND_TONE)
    _fill_rounded_rect(context, 0, 0, scale, scale, scale * 0.225)

    _set_fill(context, MARK_TONE)
    bar_width = scale * 0.038
    bar_xs = (0.245, 0.315, 0.385, 0.455, 0.525)
    bar_heights = (0.18, 0.32, 0.46, 0.30, 0.16)
    center_y = scale * 0.50
    for x_ratio, height_ratio in zip(bar_xs, bar_heights, strict=True):
        height = scale * height_ratio
        _fill_rounded_rect(
            context,
            scale * x_ratio - bar_width / 2,
            center_y - height / 2,
            bar_width,
            height,
            bar_width / 2,
        )

    # A short baseline carries the waveform into an I-beam text caret.
    _fill_rounded_rect(
        context,
        scale * 0.565,
        center_y - scale * 0.015,
        scale * 0.10,
        scale * 0.03,
        scale * 0.015,
    )
    caret_x = scale * 0.72
    caret_width = scale * 0.032
    caret_height = scale * 0.42
    cap_width = scale * 0.13
    cap_height = scale * 0.028
    _fill_rounded_rect(
        context,
        caret_x - caret_width / 2,
        center_y - caret_height / 2,
        caret_width,
        caret_height,
        caret_width / 2,
    )
    for cap_y in (
        center_y - caret_height / 2,
        center_y + caret_height / 2 - cap_height,
    ):
        _fill_rounded_rect(
            context,
            caret_x - cap_width / 2,
            cap_y,
            cap_width,
            cap_height,
            cap_height / 2,
        )

    image = CGBitmapContextCreateImage(context)
    representation = NSBitmapImageRep.alloc().initWithCGImage_(image)
    png_data = representation.representationUsingType_properties_(NSPNGFileType, None)
    png_data.writeToFile_atomically_(output_path, True)


def _set_fill(context, tone):
    CGContextSetRGBFillColor(context, *tone)


def _fill_rounded_rect(context, x, y, width, height, radius):
    radius = min(radius, width / 2, height / 2)
    CGContextBeginPath(context)
    CGContextMoveToPoint(context, x + radius, y)
    CGContextAddLineToPoint(context, x + width - radius, y)
    CGContextAddArc(
        context,
        x + width - radius,
        y + radius,
        radius,
        -1.57079632679,
        0,
        False,
    )
    CGContextAddLineToPoint(context, x + width, y + height - radius)
    CGContextAddArc(
        context,
        x + width - radius,
        y + height - radius,
        radius,
        0,
        1.57079632679,
        False,
    )
    CGContextAddLineToPoint(context, x + radius, y + height)
    CGContextAddArc(
        context,
        x + radius,
        y + height - radius,
        radius,
        1.57079632679,
        3.14159265359,
        False,
    )
    CGContextAddLineToPoint(context, x, y + radius)
    CGContextAddArc(
        context,
        x + radius,
        y + radius,
        radius,
        3.14159265359,
        4.71238898038,
        False,
    )
    CGContextFillPath(context)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "icon_512.png"
    generate_icon(512, output)
    print(f"Generated {output}")
