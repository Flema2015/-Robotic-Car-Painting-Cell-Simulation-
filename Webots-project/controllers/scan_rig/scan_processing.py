"""Image processing helpers for the fixed Webots scan rig."""

from pathlib import Path
import struct
import zlib

import numpy as np


def bgra_to_rgb(image):
    """Convert Webots' BGRA camera buffer to an RGB uint8 image."""
    bgra = np.asarray(image)
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError('Expected a camera image with shape (height, width, 4).')
    return np.ascontiguousarray(bgra[:, :, [2, 1, 0]], dtype=np.uint8)


def _rgb_to_hsv(image):
    """Convert an RGB uint8 image to HSV with hue in degrees and S/V in [0, 1]."""
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    delta = maximum - minimum

    hue = np.zeros(maximum.shape, dtype=np.float32)
    nonzero = delta > 0
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = (60.0 * ((green[red_max] - blue[red_max]) / delta[red_max])) % 360.0
    hue[green_max] = 60.0 * ((blue[green_max] - red[green_max]) / delta[green_max] + 2.0)
    hue[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)

    saturation = np.zeros(maximum.shape, dtype=np.float32)
    nonblack = maximum > 0
    saturation[nonblack] = delta[nonblack] / maximum[nonblack]
    return hue, saturation, maximum


def exclude_rgb_range(mask, rgb, minimum_rgb, maximum_rgb):
    """Remove mask pixels whose RGB channels fall inside an inclusive range."""
    binary = np.asarray(mask, dtype=bool)
    image = np.asarray(rgb)
    minimum = np.asarray(minimum_rgb, dtype=np.int16)
    maximum = np.asarray(maximum_rgb, dtype=np.int16)
    if binary.ndim != 2 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a 2D mask and an RGB image with shape (height, width, 3).')
    if binary.shape != image.shape[:2]:
        raise ValueError('Mask and RGB image must have matching height and width.')
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError('RGB range bounds must contain exactly three channels.')
    if np.any(minimum < 0) or np.any(maximum > 255) or np.any(minimum > maximum):
        raise ValueError('RGB range bounds must be ordered values between 0 and 255.')

    matches = np.all(
        (image >= minimum) & (image <= maximum),
        axis=2,
    )
    return binary & ~matches


def add_neighbor_supported_candidates(mask, candidates):
    """Add candidate pixels that touch the original mask in an 8-neighborhood."""
    binary = np.asarray(mask, dtype=bool)
    possible = np.asarray(candidates, dtype=bool)
    if binary.ndim != 2 or possible.ndim != 2:
        raise ValueError('Expected masks with shape (height, width).')
    if binary.shape != possible.shape:
        raise ValueError('Mask and candidate image must have matching height and width.')

    neighbors = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
            source_columns = slice(
                max(0, -column_offset),
                min(width, width - column_offset),
            )
            target_rows = slice(max(0, row_offset), min(height, height + row_offset))
            target_columns = slice(
                max(0, column_offset),
                min(width, width + column_offset),
            )
            neighbors[target_rows, target_columns] |= binary[source_rows, source_columns]

    return binary | (possible & neighbors)


def exclude_image_regions(mask, regions):
    """Clear rectangular image regions given as (x_min, y_min, x_max, y_max)."""
    binary = np.asarray(mask, dtype=bool).copy()
    if binary.ndim != 2:
        raise ValueError('Expected a mask with shape (height, width).')
    height, width = binary.shape
    for region in regions:
        if len(region) != 4 or any(not isinstance(value, (int, np.integer)) for value in region):
            raise ValueError('Image regions must contain four integer coordinates.')
        x_min, y_min, x_max, y_max = region
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise ValueError('Image region coordinates must be ordered and within the image.')
        binary[y_min:y_max, x_min:x_max] = False
    return binary


def retain_polygon_region(mask, polygon):
    """Keep mask pixels inside a calibrated polygon of (x, y) image coordinates."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError('Expected a mask with shape (height, width).')
    vertices = np.asarray(polygon, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError('A polygon must contain at least three (x, y) vertices.')
    if not np.all(np.isfinite(vertices)):
        raise ValueError('Polygon coordinates must be finite.')

    height, width = binary.shape
    if (
        np.any(vertices[:, 0] < 0)
        or np.any(vertices[:, 0] >= width)
        or np.any(vertices[:, 1] < 0)
        or np.any(vertices[:, 1] >= height)
    ):
        raise ValueError('Polygon coordinates must lie within the image.')

    rows, columns = np.indices(binary.shape, dtype=np.float64)
    inside = np.zeros(binary.shape, dtype=bool)
    on_boundary = np.zeros(binary.shape, dtype=bool)
    previous_x, previous_y = vertices[-1]
    for current_x, current_y in vertices:
        crossing_rows = (current_y > rows) != (previous_y > rows)
        crossing_x = (
            (previous_x - current_x)
            * (rows - current_y)
            / (previous_y - current_y + np.finfo(np.float64).eps)
            + current_x
        )
        inside ^= crossing_rows & (columns < crossing_x)

        edge_x = previous_x - current_x
        edge_y = previous_y - current_y
        relative_x = columns - current_x
        relative_y = rows - current_y
        cross = relative_x * edge_y - relative_y * edge_x
        edge_length_squared = edge_x * edge_x + edge_y * edge_y
        projection = relative_x * edge_x + relative_y * edge_y
        on_boundary |= (
            (np.abs(cross) <= 0.5 * np.sqrt(edge_length_squared))
            & (projection >= 0.0)
            & (projection <= edge_length_squared)
        )
        previous_x, previous_y = current_x, current_y

    return binary & (inside | on_boundary)


def make_red_body_mask(
    rgb,
    depth,
    floor_range_m=None,
    min_body_height_m=0.12,
    hue_low_deg=16.0,
    hue_high_deg=344.0,
    min_saturation=0.30,
    min_value=0.12,
    floor_depth_limit_m=None,
):
    """Select red, saturated pixels above the floor plane.

    The depth threshold prevents red checkerboard tiles at floor height from being
    mistaken for the car. Grey, low-saturation windows are excluded by the color test.
    """
    image = np.asarray(rgb)
    ranges = np.asarray(depth, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected an RGB image with shape (height, width, 3).')
    if ranges.shape != image.shape[:2]:
        raise ValueError('Camera and RangeFinder images must have matching height and width.')
    if not 0.0 <= hue_low_deg < hue_high_deg <= 360.0:
        raise ValueError('Hue bounds must satisfy 0 <= low < high <= 360.')
    if not 0.0 <= min_saturation <= 1.0 or not 0.0 <= min_value <= 1.0:
        raise ValueError('Saturation and value thresholds must be in [0, 1].')
    hue, saturation, value = _rgb_to_hsv(image)
    is_red = (hue <= hue_low_deg) | (hue >= hue_high_deg)
    is_color = (saturation >= min_saturation) & (value >= min_value)
    if floor_depth_limit_m is None:
        if floor_range_m is None:
            raise ValueError('Provide floor_range_m or floor_depth_limit_m.')
        if floor_range_m <= min_body_height_m:
            raise ValueError('floor_range_m must exceed min_body_height_m.')
        floor_depth_limit = float(floor_range_m) - min_body_height_m
    else:
        floor_depth_limit = np.asarray(floor_depth_limit_m, dtype=np.float32)
        if floor_depth_limit.shape != ranges.shape:
            raise ValueError('Floor-depth limits must match the depth image dimensions.')

    above_floor = (
        np.isfinite(ranges)
        & (ranges > 0.0)
        & (ranges < floor_depth_limit)
    )
    return is_red & is_color & above_floor


def floor_depth_limit_map(width, height, horizontal_fov, position, axis_angle, floor_z=0.0,
                          min_body_height_m=0.12):
    """Compute axial camera depths that correspond to points min_body_height above the floor."""
    if width <= 0 or height <= 0 or not 0.0 < horizontal_fov < np.pi:
        raise ValueError('Invalid camera dimensions or horizontal field of view.')
    if min_body_height_m < 0.0:
        raise ValueError('min_body_height_m must be non-negative.')

    axis = np.asarray(axis_angle[:3], dtype=np.float64)
    angle = float(axis_angle[3])
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0 and angle != 0.0:
        raise ValueError('A nonzero rotation angle requires a nonzero axis.')
    if axis_norm == 0.0:
        rotation = np.eye(3)
    else:
        x, y, z = axis / axis_norm
        cosine, sine = np.cos(angle), np.sin(angle)
        one_minus_cosine = 1.0 - cosine
        rotation = np.array(
            [
                [cosine + x * x * one_minus_cosine,
                 x * y * one_minus_cosine - z * sine,
                 x * z * one_minus_cosine + y * sine],
                [y * x * one_minus_cosine + z * sine,
                 cosine + y * y * one_minus_cosine,
                 y * z * one_minus_cosine - x * sine],
                [z * x * one_minus_cosine - y * sine,
                 z * y * one_minus_cosine + x * sine,
                 cosine + z * z * one_minus_cosine],
            ],
            dtype=np.float64,
        )

    focal_length_px = width / (2.0 * np.tan(horizontal_fov / 2.0))
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    horizontal = (columns - (width - 1) / 2.0) / focal_length_px
    vertical = ((height - 1) / 2.0 - rows) / focal_length_px
    ray_y, ray_z = np.meshgrid(horizontal, vertical)
    ray_z_world = rotation[2, 0] + rotation[2, 1] * ray_y + rotation[2, 2] * ray_z

    sensor_z = float(position[2])
    below_horizon = ray_z_world < -1e-9
    limit = np.full((height, width), np.inf, dtype=np.float32)
    ground_depth = (floor_z - sensor_z) / ray_z_world[below_horizon]
    height_margin_depth = min_body_height_m / -ray_z_world[below_horizon]
    limit[below_horizon] = ground_depth - height_margin_depth
    limit[limit <= 0.0] = np.inf
    return limit


def depth_preview(depth, max_range_m):
    """Create an 8-bit preview where nearer finite surfaces appear brighter."""
    ranges = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= max_range_m)
    preview = np.zeros(ranges.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(ranges[valid], [2.0, 98.0])
        if high <= low:
            preview[valid] = 255
        else:
            scaled = (high - ranges[valid]) / (high - low)
            preview[valid] = np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)
    return preview


def save_png(path, image):
    """Write an 8-bit grayscale or RGB PNG without extra imaging dependencies."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError('PNG image data must use uint8 values.')
    if array.ndim == 2:
        color_type, channels = 0, 1
    elif array.ndim == 3 and array.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise ValueError('PNG image must be grayscale or RGB.')

    height, width = array.shape[:2]
    rows = array.reshape(height, width * channels)
    raw = b''.join(b'\x00' + row.tobytes() for row in rows)

    def chunk(kind, data):
        return (
            struct.pack('>I', len(data))
            + kind
            + data
            + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack('>IIBBBBB', width, height, 8, color_type, 0, 0, 0)
    png_data = (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', header)
        + chunk(b'IDAT', zlib.compress(raw))
        + chunk(b'IEND', b'')
    )
    Path(path).write_bytes(png_data)
