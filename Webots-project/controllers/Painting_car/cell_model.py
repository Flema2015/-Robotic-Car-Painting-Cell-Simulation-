"""Modelo de la celda del robot pintor.

Lo usan Painting_car.py (dentro de Webots) y reach_study.py (fuera de Webots).

Marco del auto: origen en el CENTRO del auto sobre el piso, x hacia adelante, y hacia la
izquierda, z hacia arriba. Todo lo marcado como ESTIMACION hay que calibrarlo mirando el
contorno amarillo que dibuja Painting_car.py sobre el auto.
"""

import warnings
import numpy as np
from ikpy.chain import Chain

# ---------------- Herramienta ----------------
TOOL_TIP = np.array([0.0, 0.0, 0.26])  # punta de la boquilla en el marco del Solid GUN (m)
STANDOFF = 0.25                        # distancia boquilla-superficie al pintar (m)

# ---------------- Auto ----------------
# Del origen del nodo BmwX5 al centro del auto, en los ejes del auto. ESTIMACION.
CAR_ORIGIN_TO_CENTER = np.array([1.4, 0.0, 0.0])


def _patch(center, u, v, normal, half_u, half_v):
    """Rectangulo sobre una superficie: centro, dos ejes en el plano, normal hacia afuera y medios lados."""
    return dict(center=np.array(center, float), u=np.array(u, float), v=np.array(v, float),
                normal=np.array(normal, float), half_u=half_u, half_v=half_v)


# Parcelas a pintar (marco del auto). Las pasadas van a lo largo de u; se separan a lo largo de v.
# Calibradas con el RangeFinder superior; las trayectorias siguen pendientes de validar en Webots.
PATCHES = {
    'capo':    _patch(
        (1.65, 0.00, 0.793),
        (0.9894, 0.0, -0.1451),
        (0.0, 1.0, 0.0),
        (0.1451, 0.0, 0.9894),
        0.60,
        0.85,
    ),
    'techo':   _patch((-0.78, 0.00, 1.38), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.10, 0.65),
    'lateral': _patch((0.20, 0.93, 0.78), (1, 0, 0), (0, 0, 1), (0, 1, 0), 1.00, 0.20),
}


# ---------------- Estaciones del robot ----------------
# Una estacion es una posicion de la base del robot respecto del auto:
#   g = distancia del eje del robot a la linea central del auto (m)
#   s = corrimiento a lo largo del auto (+ hacia el frente) (m)
#   lado = L/R (costados) o F/B (frente y parte trasera)
# El robot siempre mira hacia el auto.
def station_pose(g, s, side, yaw_offset_deg=0.0):
    """Base del robot en el marco del auto: (posicion, rotacion). La rotacion lleva ejes de la base -> auto."""
    if side == 'L':
        position = np.array([s, g, 0.0])
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    elif side == 'R':
        position = np.array([s, -g, 0.0])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    elif side == 'F':
        position = np.array([g, s, 0.0])
        rotation = np.diag([-1.0, -1.0, 1.0])
    elif side == 'B':
        position = np.array([-g, s, 0.0])
        rotation = np.eye(3)
    else:
        raise ValueError("side must be 'L', 'R', 'F', or 'B'.")

    yaw = np.radians(float(yaw_offset_deg))
    c, sin_yaw = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array(
        [[c, -sin_yaw, 0.0], [sin_yaw, c, 0.0], [0.0, 0.0, 1.0]]
    )
    return position, yaw_rotation @ rotation


# Estaciones candidatas heredadas; revalidar cobertura con las parcelas calibradas.
STATIONS = [
    (1.5, -0.5, 'L', -30.0),
    (1.8, 1.0, 'L', -30.0),
    (1.8, 0.0, 'R', -30.0),
    # Complemento para el techo; requiere validacion en DRY_RUN antes de moverse.
    (1.5, -0.5, 'R', -30.0),
]


# ---------------- Cambios de coordenadas ----------------
def car_to_world(P, center_world, R_car):
    """Puntos del marco del auto -> mundo."""
    return np.asarray(P, float) @ R_car.T + center_world


def world_to_base(P, base_pos, R_base):
    """Puntos del mundo -> marco de la base del robot (el que usa ikpy)."""
    return (np.asarray(P, float) - base_pos) @ R_base


# ---------------- Trayectorias ----------------
def raster(patch, spacing=0.15, step=0.08, standoff=STANDOFF, margin=0.05):
    """Zigzag sobre una parcela, todo en el marco del auto.

    Devuelve (superficie, punta, direccion, pasada):
      superficie: punto de la superficie que se pinta
      punta:      donde va la boquilla (superficie + normal * standoff)
      direccion:  hacia donde apunta la pistola (la normal con signo contrario)
      pasada:     numero de pasada de cada punto
    """
    hu = patch['half_u'] - margin
    hv = patch['half_v'] - margin
    n_pass = max(2, int(np.floor(2 * hv / spacing)) + 1)
    n_pts = max(2, int(np.floor(2 * hu / step)) + 1)
    vs = np.linspace(-hv, hv, n_pass)
    us = np.linspace(-hu, hu, n_pts)

    S, pid = [], []
    for k, v in enumerate(vs):
        seq = us if k % 2 == 0 else us[::-1]  # zigzag: las pasadas impares van al reves
        for u in seq:
            S.append(patch['center'] + u * patch['u'] + v * patch['v'])
            pid.append(k)
    S = np.array(S)
    n = patch['normal'] / np.linalg.norm(patch['normal'])
    return S, S + standoff * n, np.tile(-n, (len(S), 1)), np.array(pid)


def paint_coverage_mask(surface_points, painted_points, radius):
    """Return which surface samples lie inside the simulated paint footprint."""
    surface = np.asarray(surface_points, dtype=float)
    painted = np.asarray(painted_points, dtype=float)
    if surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError('surface_points must have shape (n, 3).')
    if painted.size == 0:
        painted = np.empty((0, 3), dtype=float)
    if painted.ndim != 2 or painted.shape[1] != 3:
        raise ValueError('painted_points must have shape (n, 3).')
    if radius < 0.0:
        raise ValueError('Paint footprint radius must be non-negative.')
    if not len(surface) or not len(painted):
        return np.zeros(len(surface), dtype=bool)
    distances = np.linalg.norm(surface[:, None, :] - painted[None, :, :], axis=2)
    return np.any(distances <= radius + 1e-9, axis=1)


def grid(patch, nu=5, nv=5, standoff=STANDOFF, margin=0.05):
    """Grilla regular de puntos de una parcela (para el estudio de alcance)."""
    hu = patch['half_u'] - margin
    hv = patch['half_v'] - margin
    S = np.array([patch['center'] + u * patch['u'] + v * patch['v']
                  for v in np.linspace(-hv, hv, nv) for u in np.linspace(-hu, hu, nu)])
    n = patch['normal'] / np.linalg.norm(patch['normal'])
    return S, S + standoff * n, np.tile(-n, (len(S), 1))


# ---------------- Cinematica ----------------
def make_chain(urdf_path, tool_tip=TOOL_TIP):
    """Cadena de ikpy con la punta de la boquilla como ultimo eslabon. Devuelve (cadena, mascara)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        probe = Chain.from_urdf_file(urdf_path, last_link_vector=tool_tip)
    mask = ['motor' in link.name for link in probe.links]
    chain = Chain.from_urdf_file(urdf_path, last_link_vector=tool_tip, active_links_mask=mask)
    return chain, mask


def angle_deg(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def unwrap(chain, q, seed):
    """Elige, para cada articulacion, el angulo equivalente (mas o menos 2*pi) mas cercano a 'seed'.

    Dos angulos que difieren en 2*pi dejan al brazo en la misma pose, pero para los motores son
    movimientos muy distintos: sin esto el brazo puede dar una vuelta entera entre dos puntos.
    """
    q = np.array(q, float)
    for i, link in enumerate(chain.links):
        lo, hi = link.bounds if link.bounds is not None else (-np.inf, np.inf)
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        diff = (q[i] - seed[i] + np.pi) % (2 * np.pi) - np.pi
        cand = seed[i] + diff
        if lo - 1e-9 <= cand <= hi + 1e-9:
            q[i] = cand
    return q


def keep_spin(chain, q, seed):
    """Deja la ultima articulacion (el giro de la pistola sobre su propio eje) donde estaba.

    La punta de la boquilla esta sobre el eje de ese motor, asi que girarlo no cambia ni la
    posicion de la punta ni hacia donde apunta: es un grado de libertad "libre" que el solver
    usa al azar (puede dar medias vueltas de un punto al siguiente). Lo fijamos y el modelo
    verifica despues que la pose siga siendo buena.
    """
    idx = [i for i, active in enumerate(chain.active_links_mask) if active]
    if not idx:
        return q
    q = np.array(q, float)
    i6 = idx[-1]
    lo, hi = chain.links[i6].bounds if chain.links[i6].bounds is not None else (-np.inf, np.inf)
    q[i6] = min(max(float(seed[i6]), lo), hi)
    return q


def _clip_to_bounds(chain, q):
    q = np.array(q, float)
    for i, link in enumerate(chain.links):
        if link.bounds is not None:
            lo, hi = link.bounds
            if np.isfinite(lo) and np.isfinite(hi):
                q[i] = min(max(q[i], lo), hi)
    return q


def _flip_wrist(chain, q):
    """Pose equivalente de la muñeca (eje 4 + pi, eje 5 con signo cambiado, eje 6 + pi).

    En un brazo con muñeca esferica (como el IRB 4600) deja la herramienta exactamente igual,
    pero es otra configuracion para los motores. Sirve para elegir la mas cercana a la actual.
    """
    idx = [i for i, active in enumerate(chain.active_links_mask) if active]
    if len(idx) < 6:
        return None
    i4, i5, i6 = idx[-3:]
    q2 = np.array(q, float)
    q2[i4] += np.pi
    q2[i5] = -q2[i5]
    q2[i6] += np.pi
    return _clip_to_bounds(chain, q2)


def solve(chain, tip, direction, seed=None, tol_mm=10.0, tol_deg=5.0,
          candidate_validator=None):
    """Cinematica inversa de la punta con el eje z de la herramienta apuntando en 'direction'.

    Prueba varios puntos de partida: la pose anterior, su muñeca "dada vuelta" y, si ninguna
    sirve, la pose cero (un solver local puede quedarse atascado lejos de la solucion). Entre las
    soluciones buenas elige la que menos obliga a mover los motores. Si se proporciona un
    validador, prioriza las soluciones que lo superan antes de minimizar el movimiento.
    Devuelve (angulos, error de posicion en mm, error de orientacion en grados), segun el MODELO.
    """
    def attempt(s0):
        q = chain.inverse_kinematics(tip, target_orientation=direction, orientation_mode='Z',
                                     initial_position=s0)
        if seed is not None:
            q = unwrap(chain, q, seed)
            q = keep_spin(chain, q, seed)
        fk = chain.forward_kinematics(q)
        e = float(np.linalg.norm(fk[:3, 3] - tip) * 1000)
        a = angle_deg(fk[:3, 2], direction)
        ok = e <= tol_mm and a <= tol_deg
        safe = ok and (candidate_validator is None or candidate_validator(q))
        move = 0.0 if seed is None else float(np.max(np.abs(q - np.array(seed, float))))
        if candidate_validator is None:
            score = (not ok, move if ok else e)
        else:
            score = (not ok, not safe, move if ok else e)
        return score, q, e, a

    seeds = [seed]
    if seed is not None:
        flipped = _flip_wrist(chain, seed)
        if flipped is not None:
            seeds.append(flipped)
        if candidate_validator is not None:
            seeds.append(None)

    best = None
    for s0 in seeds:
        cand = attempt(s0)
        if best is None or cand[0] < best[0]:
            best = cand
    if best[0][0] and seed is not None:   # ninguna fue buena: ultimo recurso, partir de la pose cero
        cand = attempt(None)
        if cand[0] < best[0]:
            best = cand
    return best[1], best[2], best[3]


def plan_patch(chain, patch, car_center, car_R, base_pos, base_R, seed,
               spacing=0.15, step=0.08, pos_tol_mm=10.0, ang_tol_deg=5.0,
               jump_max_rad=1.0, candidate_validator=None, progress_callback=None):
    """Planifica el zigzag de una parcela desde la posicion actual del robot.

    Resuelve la cinematica inversa punto por punto, partiendo de la solucion anterior. Devuelve
    todo lo necesario para recorrerlo: puntos en el mundo y en el marco de la base, angulos,
    cuales son alcanzables y los errores del modelo.
    """
    S, tip, direc, pid = raster(patch, spacing, step)

    S_w = car_to_world(S, car_center, car_R)
    tip_w = car_to_world(tip, car_center, car_R)
    dir_w = direc @ car_R.T

    tip_b = world_to_base(tip_w, base_pos, base_R)
    dir_b = dir_w @ base_R

    qs, errs, angs, oks, safe_points, brk = [], [], [], [], [], []
    q = seed
    prev_q, prev_ok = None, False
    total_points = len(tip_b)
    for point_index, (p, d) in enumerate(zip(tip_b, dir_b), start=1):
        q_new, e, a = solve(
            chain,
            p,
            d,
            q,
            pos_tol_mm,
            ang_tol_deg,
            candidate_validator=candidate_validator,
        )
        ok = e <= pos_tol_mm and a <= ang_tol_deg
        safe = ok and (candidate_validator is None or candidate_validator(q_new))
        # Un salto grande de articulaciones respecto del punto anterior = cambio de configuracion del brazo.
        # No lo recorremos pegado a la superficie: cortamos el tramo ahi (se retira y se vuelve a acercar).
        jump = prev_ok and safe and float(np.max(np.abs(np.array(q_new) - np.array(prev_q)))) > jump_max_rad
        if safe:
            q = q_new  # solo seguimos desde soluciones buenas
        qs.append(q_new)
        errs.append(e)
        angs.append(a)
        oks.append(ok)
        safe_points.append(safe)
        brk.append(bool(jump))
        prev_q, prev_ok = q_new, safe
        if progress_callback is not None and (
            point_index == 1 or point_index % 25 == 0 or point_index == total_points
        ):
            progress_callback(point_index, total_points)

    return dict(S_w=S_w, tip_w=tip_w, dir_w=dir_w, tip_b=tip_b, dir_b=dir_b, pid=pid,
                q=qs, ok=np.array(oks), safe=np.array(safe_points), brk=np.array(brk),
                err_mm=np.array(errs), err_deg=np.array(angs))