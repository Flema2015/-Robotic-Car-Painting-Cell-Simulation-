"""Capture fixed RGB-D views and segment visible red car-body pixels."""

from pathlib import Path

import numpy as np
from controller import Supervisor

from scan_processing import (
    bgra_to_rgb,
    add_neighbor_supported_candidates,
    depth_preview,
    exclude_image_regions,
    exclude_rgb_range,
    floor_depth_limit_map,
    make_red_body_mask,
    retain_polygon_region,
    save_png,
)

OUTPUT_DIR = Path(__file__).resolve().parent / 'scan_output'
SENSOR_WARMUP_STEPS = 5
MIN_BODY_HEIGHT_M = 0.12

RED_HUE_LOW_DEG = 16.0
RED_HUE_HIGH_DEG = 344.0
RED_MIN_SATURATION = 0.30
RED_MIN_VALUE = 0.12
NEIGHBOR_HUE_LOW_DEG = 28.0
NEIGHBOR_HUE_HIGH_DEG = 332.0
NEIGHBOR_MIN_SATURATION = 0.18
NEIGHBOR_MIN_VALUE = 0.07
ARM_RGB_MIN = (50, 30, 20)
ARM_RGB_MAX = (85, 60, 60)
# In this fixed view, the visible robot arm is above and left of the car silhouette.
ARM_IMAGE_REGIONS = {
    'side_nx': ((0, 0, 140, 85),),
}
END_CAR_POLYGONS = {
    'end_py': (
        (82, 99), (181, 99), (198, 105), (211, 118), (219, 137),
        (222, 163), (219, 186), (213, 202), (201, 213), (104, 213),
        (91, 204), (84, 187), (82, 155), (89, 132), (98, 114),
    ),
    'end_ny': (
        (101, 99), (175, 99), (190, 105), (202, 115), (210, 130),
        (224, 146), (238, 164), (240, 178), (236, 192), (226, 204),
        (211, 212), (126, 214), (112, 208), (104, 197), (101, 182),
        (102, 164), (108, 145), (115, 129), (122, 113),
    ),
}

# World-space poses; keep these synchronized with the sensor nodes in the .wbt.
VIEWS = {
    'top': {
        'camera': 'cam_top',
        'range_finder': 'rf_top',
        'position': (-0.002, -0.395, 6.0),
        'rotation': (0.0, 1.0, 0.0, 1.5708),
    },
    'side_px': {
        'camera': 'cam_side_px',
        'range_finder': 'rf_side_px',
        'position': (3.0, -0.395, 1.5),
        'rotation': (-0.090476903, 0.0, 0.995898554, 3.141592654),
    },
    'side_nx': {
        'camera': 'cam_side_nx',
        'range_finder': 'rf_side_nx',
        'position': (-3.0, -0.395, 1.5),
        'rotation': (0.0, 1.0, 0.0, 0.181438099),
    },
    'end_py': {
        'camera': 'cam_end_py',
        'range_finder': 'rf_end_py',
        'position': (-0.002, 3.4, 1.45),
        'rotation': (0.0, 0.130623459, -0.991432051, 1.570796327),
    },
    'end_ny': {
        'camera': 'cam_end_ny',
        'range_finder': 'rf_end_ny',
        'position': (-0.002, -4.2, 1.45),
        'rotation': (0.0, 0.130285999, 0.991476454, 1.570796327),
    },
}

robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
sensors = {}
for view_name, config in VIEWS.items():
    camera = robot.getDevice(config['camera'])
    range_finder = robot.getDevice(config['range_finder'])
    if camera is None or range_finder is None:
        raise RuntimeError(
            f'Faltan los sensores de la vista {view_name}: '
            f'{config["camera"]}, {config["range_finder"]}.'
        )
    if camera.getWidth() != range_finder.getWidth() or camera.getHeight() != range_finder.getHeight():
        raise RuntimeError(f'Camera y RangeFinder de {view_name} tienen resoluciones distintas.')
    camera.enable(time_step)
    range_finder.enable(time_step)
    sensors[view_name] = (camera, range_finder)

for _ in range(SENSOR_WARMUP_STEPS):
    if robot.step(time_step) == -1:
        raise SystemExit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for view_name, config in VIEWS.items():
    camera, range_finder = sensors[view_name]
    width, height = camera.getWidth(), camera.getHeight()
    bgra = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((height, width, 4))
    rgb = bgra_to_rgb(bgra)
    depth = np.asarray(range_finder.getRangeImage(), dtype=np.float32).reshape((height, width))
    floor_limit = floor_depth_limit_map(
        width,
        height,
        camera.getFov(),
        config['position'],
        config['rotation'],
        min_body_height_m=MIN_BODY_HEIGHT_M,
    )
    red_mask = make_red_body_mask(
        rgb,
        depth,
        floor_depth_limit_m=floor_limit,
        hue_low_deg=RED_HUE_LOW_DEG,
        hue_high_deg=RED_HUE_HIGH_DEG,
        min_saturation=RED_MIN_SATURATION,
        min_value=RED_MIN_VALUE,
    )
    recovered_neighbor_pixels = 0
    excluded_arm_pixels = 0
    if view_name in ('side_px', 'side_nx'):
        red_mask_without_arm = exclude_rgb_range(
            red_mask,
            rgb,
            ARM_RGB_MIN,
            ARM_RGB_MAX,
        )
        excluded_arm_pixels = int(red_mask.sum() - red_mask_without_arm.sum())
        red_mask = red_mask_without_arm
        neighbor_candidates = make_red_body_mask(
            rgb,
            depth,
            floor_depth_limit_m=floor_limit,
            hue_low_deg=NEIGHBOR_HUE_LOW_DEG,
            hue_high_deg=NEIGHBOR_HUE_HIGH_DEG,
            min_saturation=NEIGHBOR_MIN_SATURATION,
            min_value=NEIGHBOR_MIN_VALUE,
        )
        neighbor_candidates = exclude_rgb_range(
            neighbor_candidates,
            rgb,
            ARM_RGB_MIN,
            ARM_RGB_MAX,
        )
        if view_name in ARM_IMAGE_REGIONS:
            neighbor_candidates = exclude_image_regions(
                neighbor_candidates,
                ARM_IMAGE_REGIONS[view_name],
            )
        red_mask_with_neighbors = add_neighbor_supported_candidates(
            red_mask,
            neighbor_candidates,
        )
        recovered_neighbor_pixels = int(
            red_mask_with_neighbors.sum() - red_mask.sum()
        )
        red_mask = red_mask_with_neighbors
    if view_name in ARM_IMAGE_REGIONS:
        red_mask_without_arm = exclude_image_regions(
            red_mask,
            ARM_IMAGE_REGIONS[view_name],
        )
        excluded_arm_pixels += int(red_mask.sum() - red_mask_without_arm.sum())
        red_mask = red_mask_without_arm
    if view_name in END_CAR_POLYGONS:
        red_mask_with_car_roi = retain_polygon_region(
            red_mask,
            END_CAR_POLYGONS[view_name],
        )
        excluded_arm_pixels += int(red_mask.sum() - red_mask_with_car_roi.sum())
        red_mask = red_mask_with_car_roi
    suffix = '' if view_name == 'top' else f'_{view_name}'
    save_png(OUTPUT_DIR / f'scan{suffix}_rgb.png', rgb)
    save_png(OUTPUT_DIR / f'scan{suffix}_red_mask.png', red_mask.astype(np.uint8) * 255)
    save_png(
        OUTPUT_DIR / f'scan{suffix}_depth.png',
        depth_preview(depth, range_finder.getMaxRange()),
    )

    overlay = rgb.copy()
    overlay[red_mask] = (
        0.45 * overlay[red_mask] + 0.55 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)
    save_png(OUTPUT_DIR / f'scan{suffix}_overlay.png', overlay)
    np.save(OUTPUT_DIR / f'scan{suffix}_depth_m.npy', depth)
    np.save(OUTPUT_DIR / f'scan{suffix}_red_mask.npy', red_mask)

    valid_depth = (
        np.isfinite(depth)
        & (depth >= range_finder.getMinRange())
        & (depth <= range_finder.getMaxRange())
    )
    valid_values = depth[valid_depth]
    range_summary = (
        f'{valid_values.min():.3f}..{valid_values.max():.3f} m'
        if valid_values.size else 'ninguna lectura valida'
    )
    print(
        f'{view_name}: RGB {width}x{height}; '
        f'rangos validos {valid_values.size}/{depth.size}; '
        f'rango {range_summary}; '
        f'mascara roja {int(red_mask.sum())}/{red_mask.size} '
        f'({100 * red_mask.mean():.1f}%); '
        f'pixeles recuperados por vecinos {recovered_neighbor_pixels}; '
        f'pixeles similares al brazo excluidos {excluded_arm_pixels}.'
    )

print(f'Resultados guardados en: {OUTPUT_DIR}')

while robot.step(time_step) != -1:
    pass
