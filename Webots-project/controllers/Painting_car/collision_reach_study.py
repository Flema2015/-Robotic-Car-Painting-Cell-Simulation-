"""Search robot stations for reachable painting points that pass the 10 cm margin.

Run outside Webots from this directory after Painting_car.py exports the URDF:
    python collision_reach_study.py
"""

import csv
import itertools
from pathlib import Path

import numpy as np

from cell_model import (
    PATCHES,
    STANDOFF,
    make_chain,
    plan_patch,
    solve,
    station_pose,
)
from collision_check import (
    BMW_X5_BODY_BOXES,
    clearance_to_vehicle,
    load_collision_boxes,
)

HERE = Path(__file__).resolve().parent
URDF = HERE / 'irb4600.urdf'
OUTPUT_CSV = HERE / 'collision_reach_study.csv'
PATCH_NAMES = ('capo', 'techo', 'lateral')
MAX_STATIONS = 3
TARGET_COVERAGE_PCT = 90.0
MARGIN_M = 0.10
TOOL_RADIUS_M = 0.035
SPACING_M = 0.15
STEP_M = 0.08
APPROACH_M = 0.05
STOW_TIP = np.array([0.1, 0.0, 2.2])
STOW_DIR = np.array([0.0, 0.0, 1.0])
POSITION_TOLERANCE_MM = 10.0
ANGLE_TOLERANCE_DEG = 5.0
GAPS_M = (1.5, 1.8, 2.1)
SHIFTS_M = (-1.0, -0.5, 0.0, 0.5, 1.0)
SIDES = ('L', 'R', 'F', 'B')
YAW_OFFSETS_DEG = (-30.0, 0.0, 30.0)
COARSE_GRID_SIZE = 3


def _clearance(chain, boxes, joints, position, rotation):
    return clearance_to_vehicle(
        chain,
        joints,
        boxes,
        position,
        rotation,
        np.zeros(3),
        np.eye(3),
        BMW_X5_BODY_BOXES,
        margin_m=MARGIN_M,
        tool_radius_m=TOOL_RADIUS_M,
    )


def _station_label(station):
    gap, shift, side, yaw = station
    return f'(g={gap:.1f}, s={shift:+.1f}, {side}, yaw={yaw:+.0f} deg)'


def _score_combo(combo, flags):
    coverage = {
        name: float(np.logical_or.reduce([flags[index][name] for index in combo]).mean() * 100.0)
        for name in PATCH_NAMES
    }
    return (min(coverage.values()), float(np.mean(list(coverage.values())))), coverage


def _best_combinations(flags, max_stations):
    candidates = range(len(flags))
    best_by_count = {}
    for count in range(1, max_stations + 1):
        best = None
        for combo in itertools.combinations(candidates, count):
            score, coverage = _score_combo(combo, flags)
            if best is None or score > best[0]:
                best = score, combo, coverage
        best_by_count[count] = best
    return best_by_count


def _path_is_clear(chain, boxes, start_q, target_q, position, rotation):
    start = np.asarray(start_q, dtype=float)
    target = np.asarray(target_q, dtype=float)
    steps = max(1, int(np.ceil(np.max(np.abs(target - start)) / 0.05)))
    for index in range(1, steps + 1):
        sample = start + (target - start) * (index / steps)
        safe, _, _ = _clearance(chain, boxes, sample, position, rotation)
        if not safe:
            return False
    return True


def _evaluate_coarse_station(chain, boxes, station, point_grids):
    gap, shift, side, yaw = station
    base_position, base_rotation = station_pose(gap, shift, side, yaw)
    safe_by_patch = {}
    reachable_by_patch = {}
    seed = np.zeros(len(chain.links))

    for name, (_, tips, directions) in point_grids.items():
        tips_base = (tips - base_position) @ base_rotation
        directions_base = directions @ base_rotation
        q = seed.copy()
        reachable = np.zeros(len(tips), dtype=bool)
        safe_points = np.zeros(len(tips), dtype=bool)
        for index, (tip, direction) in enumerate(zip(tips_base, directions_base)):
            q_new, position_error, angle_error = solve(
                chain,
                tip,
                direction,
                q,
                POSITION_TOLERANCE_MM,
                ANGLE_TOLERANCE_DEG,
                candidate_validator=lambda candidate: _clearance(
                    chain, boxes, candidate, base_position, base_rotation
                )[0],
            )
            reachable[index] = (
                position_error <= POSITION_TOLERANCE_MM
                and angle_error <= ANGLE_TOLERANCE_DEG
            )
            if reachable[index]:
                safe_points[index] = _clearance(
                    chain, boxes, q_new, base_position, base_rotation
                )[0]
                if safe_points[index]:
                    q = q_new
        safe_by_patch[name] = safe_points
        reachable_by_patch[name] = reachable

    return safe_by_patch, reachable_by_patch


def _evaluate_dense_station(chain, boxes, station):
    gap, shift, side, yaw = station
    base_position, base_rotation = station_pose(gap, shift, side, yaw)
    output = {}
    seed = np.zeros(len(chain.links))
    stow_q, stow_error, stow_angle = solve(
        chain,
        STOW_TIP,
        STOW_DIR,
        seed,
        30.0,
        15.0,
        candidate_validator=lambda candidate: _clearance(
            chain, boxes, candidate, base_position, base_rotation
        )[0],
    )
    stow_safe, _, _ = _clearance(
        chain, boxes, stow_q, base_position, base_rotation
    )

    for name in PATCH_NAMES:
        plan = plan_patch(
            chain,
            PATCHES[name],
            np.zeros(3),
            np.eye(3),
            base_position,
            base_rotation,
            stow_q,
            SPACING_M,
            STEP_M,
            POSITION_TOLERANCE_MM,
            ANGLE_TOLERANCE_DEG,
            candidate_validator=lambda candidate: _clearance(
                chain, boxes, candidate, base_position, base_rotation
            )[0],
        )
        safe_path = np.zeros(len(plan['ok']), dtype=bool)
        for start, end in _segments(plan['ok'] & plan['safe'], plan['brk']):
            approach_tip = plan['tip_b'][start] - plan['dir_b'][start] * APPROACH_M
            approach_q, position_error, angle_error = solve(
                chain,
                approach_tip,
                plan['dir_b'][start],
                stow_q,
                POSITION_TOLERANCE_MM,
                ANGLE_TOLERANCE_DEG,
                candidate_validator=lambda candidate: _clearance(
                    chain, boxes, candidate, base_position, base_rotation
                )[0],
            )
            approach_ok = (
                stow_error <= 30.0
                and stow_angle <= 15.0
                and stow_safe
                and position_error <= POSITION_TOLERANCE_MM
                and angle_error <= ANGLE_TOLERANCE_DEG
                and _path_is_clear(
                    chain, boxes, stow_q, approach_q, base_position, base_rotation
                )
                and _path_is_clear(
                    chain,
                    boxes,
                    approach_q,
                    plan['q'][start],
                    base_position,
                    base_rotation,
                )
            )
            if not approach_ok:
                continue
            safe_path[start] = True
            previous_q = np.asarray(plan['q'][start], dtype=float)
            for index in range(start + 1, end + 1):
                if _path_is_clear(
                    chain,
                    boxes,
                    previous_q,
                    plan['q'][index],
                    base_position,
                    base_rotation,
                ):
                    safe_path[index] = True
                    previous_q = np.asarray(plan['q'][index], dtype=float)
        output[name] = {
            'reachable': plan['ok'],
            'safe': safe_path,
            'total': len(plan['ok']),
        }
    return output


def _segments(mask, breaks):
    segments = []
    start = None
    for index, eligible in enumerate(mask):
        if eligible and start is not None and breaks[index]:
            segments.append((start, index - 1))
            start = None
        if eligible and start is None:
            start = index
        elif not eligible and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def main():
    if not URDF.is_file():
        raise FileNotFoundError(
            f'No se encuentra {URDF}. Corre Painting_car.py en Webots para exportarlo.'
        )

    chain, _ = make_chain(str(URDF))
    boxes = load_collision_boxes(URDF, chain)
    coarse_grids = {
        name: PATCHES[name] for name in PATCH_NAMES
    }
    point_grids = {}
    from cell_model import grid

    for name, patch in coarse_grids.items():
        point_grids[name] = grid(patch, COARSE_GRID_SIZE, COARSE_GRID_SIZE, STANDOFF)

    stations = [
        (gap, shift, side, yaw)
        for side in SIDES
        for gap in GAPS_M
        for shift in SHIFTS_M
        for yaw in YAW_OFFSETS_DEG
    ]
    safe_flags = []
    reachable_flags = []
    rows = []
    print(
        f'Evaluando {len(stations)} estaciones x '
        f'{COARSE_GRID_SIZE ** 2 * len(PATCH_NAMES)} puntos coarse; '
        'cada punto busca soluciones IK que respeten el margen.'
    )
    for index, station in enumerate(stations, start=1):
        safe, reachable = _evaluate_coarse_station(chain, boxes, station, point_grids)
        safe_flags.append(safe)
        reachable_flags.append(reachable)
        rows.append(station + tuple(
            round(100.0 * safe[name].mean(), 1) for name in PATCH_NAMES
        ) + tuple(
            round(100.0 * reachable[name].mean(), 1) for name in PATCH_NAMES
        ))
        print(
            f'[{index:03d}/{len(stations)}] {_station_label(station)} -> '
            + ' '.join(
                f'{name} {100.0 * safe[name].mean():.0f}%'
                for name in PATCH_NAMES
            ),
            flush=True,
        )

    best_by_count = _best_combinations(safe_flags, MAX_STATIONS)
    for count, (_, combo, coverage) in best_by_count.items():
        print(
            f'\nMejor conjunto coarse de {count} estacion(es): '
            + ' + '.join(_station_label(stations[index]) for index in combo)
        )
        print('  ' + ' | '.join(f'{name}: {coverage[name]:.1f}%' for name in PATCH_NAMES))

    chosen_count = next(
        (
            count for count, (score, _, _) in best_by_count.items()
            if score[0] >= TARGET_COVERAGE_PCT
        ),
        MAX_STATIONS,
    )
    chosen_indices = best_by_count[chosen_count][1]
    chosen = [stations[index] for index in chosen_indices]
    print(
        f'\nValidacion densa con {chosen_count} estacion(es): '
        + ' + '.join(_station_label(station) for station in chosen)
    )

    dense = [_evaluate_dense_station(chain, boxes, station) for station in chosen]
    for name in PATCH_NAMES:
        safe_union = np.logical_or.reduce([item[name]['safe'] for item in dense])
        reachable_union = np.logical_or.reduce([item[name]['reachable'] for item in dense])
        print(
            f'  {name:8s}: IK {int(reachable_union.sum())}/{len(safe_union)} '
            f'({100.0 * reachable_union.mean():.1f}%); con margen y tramo '
            f'{int(safe_union.sum())}/{len(safe_union)} '
            f'({100.0 * safe_union.mean():.1f}%)'
        )

    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ['g_m', 's_m', 'side', 'yaw_offset_deg']
            + [f'{name}_safe_coarse_pct' for name in PATCH_NAMES]
            + [f'{name}_ik_coarse_pct' for name in PATCH_NAMES]
        )
        writer.writerows(rows)
    print(f'\nResultados coarse guardados en {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
