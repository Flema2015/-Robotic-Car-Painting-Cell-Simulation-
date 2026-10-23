"""Dia 2: estudio de alcance. Se corre FUERA de Webots, en una terminal, parado en esta carpeta:

    python reach_study.py

Necesita irb4600.urdf (lo guarda Painting_car.py al correr en Webots, DESPUES de llevar el brazo a la pose cero;
un URDF exportado con el brazo en otra pose da limites de articulacion corridos y un alcance distinto).

Idea: para cada estacion candidata del robot (a la izquierda o a la derecha del auto), probar una
grilla de puntos sobre cada parcela con la pistola perpendicular a la superficie, y marcar cuales
llegan. Despues buscar el menor conjunto de estaciones que cubre todo. Asi elegimos con datos.
"""

import os
import sys
import csv
import time
import itertools
import numpy as np

from cell_model import PATCHES, make_chain, grid, solve, station_pose

URDF = 'irb4600.urdf'

GAPS = [1.5, 1.8, 2.1]
SHIFTS = [-1.0, -0.5, 0.0, 0.5, 1.0]
SIDES = ['L', 'R']
GRIDS = {'capo': (7, 7), 'techo': (7, 5), 'lateral': (9, 3)}  # puntos por parcela (u, v)
POS_TOL_MM, ANG_TOL_DEG = 10.0, 5.0
TARGET_PCT = 90.0   # cobertura minima que buscamos en cada parcela
MAX_STATIONS = 3

if not os.path.exists(URDF):
    sys.exit(f'No encuentro {URDF}. Corre Painting_car.py una vez en Webots: lo guarda en la carpeta del controlador.')

chain, mask = make_chain(URDF)
points = {name: grid(PATCHES[name], *GRIDS[name]) for name in GRIDS}
stations = [(g, s, side) for side in SIDES for g in GAPS for s in SHIFTS]
print(f'{len(stations)} estaciones candidatas x {sum(len(v[0]) for v in points.values())} puntos. Puede tardar un par de minutos...\n')

flags = {}  # estacion -> {parcela: banderas por punto}
t0 = time.time()
for st in stations:
    g, s, side = st
    base, R = station_pose(g, s, side)
    flags[st] = {}
    for name, (S, tip, direc) in points.items():
        tip_b = (tip - base) @ R
        dir_b = direc @ R
        q, fl = None, []
        for p, d in zip(tip_b, dir_b):
            q_new, e, a = solve(chain, p, d, q, POS_TOL_MM, ANG_TOL_DEG)
            good = e <= POS_TOL_MM and a <= ANG_TOL_DEG
            if not good:  # segundo intento desde la pose cero
                q_new, e, a = solve(chain, p, d, None, POS_TOL_MM, ANG_TOL_DEG)
                good = e <= POS_TOL_MM and a <= ANG_TOL_DEG
            if good:
                q = q_new
            fl.append(good)
        flags[st][name] = np.array(fl)
    print(f'  g={g:.1f} s={s:+.1f} {side}  ' + '  '.join(f'{n}: {100 * flags[st][n].mean():5.1f}%' for n in GRIDS), flush=True)
print(f'\n(tardo {time.time() - t0:.0f} s)')


def coverage(combo):
    out = {}
    for name in GRIDS:
        u = np.zeros(len(points[name][0]), bool)
        for st in combo:
            u |= flags[st][name]
        out[name] = 100.0 * u.mean()
    return out


print('\nMEJOR CONJUNTO DE ESTACIONES SEGUN LA CANTIDAD')
best_by_k = {}
for k in range(1, MAX_STATIONS + 1):
    best = None
    for combo in itertools.combinations(stations, k):
        c = coverage(combo)
        key = (min(c.values()), float(np.mean(list(c.values()))))
        if best is None or key > best[0]:
            best = (key, combo, c)
    best_by_k[k] = best
    lab = ' + '.join(f'(g={g}, s={s:+.1f}, {side})' for g, s, side in best[1])
    print(f'  {k} estacion(es): {lab}')
    print('       -> ' + '  '.join(f'{n} {v:.0f}%' for n, v in best[2].items()))

chosen_k = next((k for k in range(1, MAX_STATIONS + 1) if best_by_k[k][0][0] >= TARGET_PCT), MAX_STATIONS)
chosen = best_by_k[chosen_k][1]
print(f'\nRECOMENDACION (menor cantidad que llega a {TARGET_PCT:.0f}% en cada parcela, o la mejor con {MAX_STATIONS}):')
print('  STATIONS = [' + ', '.join(f"({g}, {s}, '{side}')" for g, s, side in chosen) + ']')
print('  (copialo en cell_model.py)')

with open('reach_study.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['g_m', 's_m', 'lado'] + list(GRIDS))
    for st in stations:
        w.writerow([st[0], st[1], st[2]] + [round(100 * flags[st][n].mean(), 1) for n in GRIDS])
print('Guardado: reach_study.csv')

# Figura opcional: vista superior con las estaciones elegidas y lo que alcanza el conjunto.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (S, tip, direc) in points.items():
        u = np.zeros(len(S), bool)
        for st in chosen:
            u |= flags[st][name]
        for ax, (i, j) in zip(axes, [(0, 1), (0, 2)]):
            ax.scatter(tip[u, i], tip[u, j], c='tab:green', s=18, label=f'{name}: alcanza' if ax is axes[0] else None)
            ax.scatter(tip[~u, i], tip[~u, j], c='tab:red', s=22, marker='x', label=f'{name}: no alcanza' if ax is axes[0] else None)
    for g, s, side in chosen:
        axes[0].plot([s], [g if side == 'L' else -g], 'k^', ms=10)
        axes[0].annotate(f'{side} s={s:+.1f}', (s, g if side == 'L' else -g), textcoords='offset points', xytext=(6, 6), fontsize=8)
    axes[0].set_xlabel('x del auto (m)'); axes[0].set_ylabel('y del auto (m)'); axes[0].set_title('Vista superior (triangulos = estaciones del robot)')
    axes[1].set_xlabel('x del auto (m)'); axes[1].set_ylabel('z (m)'); axes[1].set_title('Vista lateral')
    for ax in axes:
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    cov = coverage(chosen)
    fig.suptitle(f'{chosen_k} estacion(es): ' + ', '.join(f'{n} {v:.0f}%' for n, v in cov.items()))
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig('reach_study.png', dpi=130)
    print('Guardado: reach_study.png')
except ImportError:
    print('(sin matplotlib: no genero la figura)')