"""Dias 2-3: recorrer en zigzag las parcelas del auto, desde varias estaciones, con la pistola perpendicular.

Requisitos en el mundo:
  - robot con supervisor TRUE y un Solid con DEF GUN dentro de handSlot (la pistola cuelga de el)
  - el nodo BmwX5 con DEF CAR (girado para que +y del auto mire al lado izquierdo; ver README del chat)
  - cell_model.py en esta misma carpeta

Con RUN_MODE = 'dry_run', dibuja las parcelas, estaciones, rutas previstas y waypoints que pasan
el margen de colision. Con RUN_MODE = 'motion_test', mueve el brazo lentamente en las estaciones
seleccionadas, sin activar pintura. Con RUN_MODE = 'paint_test', deposita marcas visuales y mide
cobertura por la huella simulada, sin modificar la textura del auto ni activar pintura fisica.
El movimiento sigue limitado por el margen de 10 cm y la deteccion de contacto de Webots.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
from controller import Supervisor

from collision_check import (
    BMW_X5_BODY_BOXES,
    clearance_to_vehicle,
    load_collision_boxes,
)

try:
    from cell_model import (PATCHES, STATIONS, TOOL_TIP, CAR_ORIGIN_TO_CENTER, make_chain,
                                car_to_world, station_pose, plan_patch, solve, raster,
                                paint_coverage_mask)
except ImportError as e:
    sys.exit(f'Falta cell_model.py en la carpeta del controlador, o ikpy: {e}')

# ---------------- Parametros ----------------
RUN_MODE = 'paint_test'  # 'dry_run', 'motion_test' o 'paint_test' (marcas visuales solamente).
MOTION_TEST_STATION_INDICES = (2, 3, 1, 4)
MOTION_TEST_PATCH = 'capo'
DRY_RUN = RUN_MODE == 'dry_run'
ALL_PATCHES = ('capo', 'techo', 'lateral')
SIMULATED_PAINT_COLOR = (0.05, 0.20, 0.95)
SIMULATED_PAINT_RADIUS_M = 0.075
if RUN_MODE not in ('dry_run', 'motion_test', 'paint_test'):
    sys.exit("RUN_MODE debe ser 'dry_run', 'motion_test' o 'paint_test'.")
if DRY_RUN:
    STATIONS_TO_RUN = list(enumerate(STATIONS, start=1))
    PATCHES_TO_PAINT = list(ALL_PATCHES)
else:
    if (
        not MOTION_TEST_STATION_INDICES
        or len(set(MOTION_TEST_STATION_INDICES)) != len(MOTION_TEST_STATION_INDICES)
        or any(not 1 <= index <= len(STATIONS) for index in MOTION_TEST_STATION_INDICES)
    ):
        sys.exit('MOTION_TEST_STATION_INDICES debe contener indices validos y no repetidos.')
    if MOTION_TEST_PATCH not in ALL_PATCHES:
        sys.exit(f'MOTION_TEST_PATCH debe ser uno de: {", ".join(ALL_PATCHES)}.')
    STATIONS_TO_RUN = [
        (index, STATIONS[index - 1])
        for index in MOTION_TEST_STATION_INDICES
    ]
    PATCHES_TO_PAINT = [MOTION_TEST_PATCH]
SPACING = 0.15          # separacion entre pasadas (m)
STEP = 0.08             # separacion entre puntos de una pasada (m)
APPROACH = 0.05         # distancia extra para acercarse y alejarse de cada tramo (m)
COLLISION_MARGIN_M = 0.10
COLLISION_SAMPLE_STEP_RAD = 0.05
PAINT_TOOL_RADIUS_M = 0.035
POS_TOL_MM = 10.0       # el modelo debe llegar con este error para considerar alcanzable un punto
ANG_TOL_DEG = 5.0
MAX_VELOCITY = 1.5      # rad/s (se limita al maximo que permite cada motor)
MOTION_TEST_MAX_VELOCITY = 0.30
REACH_TOL_RAD = 0.005   # error de articulacion con el que damos por llegado un punto
WAYPOINT_TIMEOUT_S = 4.0
STOW_TIP = np.array([0.1, 0.0, 2.2])    # pose de "guardado": brazo vertical sobre la base (marco de la base)
STOW_DIR = np.array([0.0, 0.0, 1.0])
STATION_COLORS = [(1, 0, 1), (1, 0.5, 0), (0.6, 0.3, 1)]
MOTOR_VELOCITY_LIMIT = (
    MOTION_TEST_MAX_VELOCITY if RUN_MODE in ('motion_test', 'paint_test') else MAX_VELOCITY
)

# ---------------- Webots ----------------
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
arm = robot.getSelf()

if not np.all(np.isfinite(np.array(arm.getPosition(), dtype=float))):
    sys.exit('Este robot NO es Supervisor. Pone supervisor TRUE en el robot, '
             'guarda el mundo con el boton de guardar de la barra superior y '
             'recarga el mundo (Reload World).')

gun = robot.getFromDef('GUN')
if gun is None:
    sys.exit('No encuentro el DEF GUN. Selecciona el Solid de handSlot y escribi GUN en el campo DEF.')
car = robot.getFromDef('CAR')
if car is None:
    sys.exit('No encuentro el DEF CAR. Selecciona el nodo BmwX5 en el arbol y escribi CAR en el campo DEF.')
arm.enableContactPointsTracking(time_step, True)

def step_ok():
    return robot.step(time_step) != -1


def car_contact_detected():
    car_id = car.getId()
    for point in arm.getContactPoints(includeDescendants=True):
        other = robot.getFromId(point.getNodeId())
        while other is not None:
            if other.getId() == car_id:
                return True
            parent = other.getParentNode()
            if parent is not None and parent.getId() == other.getId():
                break
            other = parent
    return False


def wait(seconds):
    for _ in range(max(1, int(seconds * 1000 / time_step))):
        if not step_ok():
            sys.exit(0)


clearance_cache = {}


def _clearance_cache_key(q, base_pos, base_R):
    return (
        np.asarray(q, dtype=np.float64).tobytes(),
        np.asarray(base_pos, dtype=np.float64).tobytes(),
        np.asarray(base_R, dtype=np.float64).tobytes(),
    )


# ---------------- Pose cero ANTES de exportar el URDF ----------------
# Webots exporta el URDF con el brazo en la pose en que esta en ese momento, y el modelo toma esa pose
# como "angulo 0". Si el mundo se guardo con el brazo en otra pose, el angulo 0 del modelo y el de los
# motores dejan de coincidir (descubierto con el diagnostico: ~0,8 m de diferencia). Por eso llevamos
# primero los motores a 0 y recien despues exportamos.
HOME_MOTORS = ['A motor', 'B motor', 'C motor', 'D motor', 'E motor', 'F motor']
home = []
for name in HOME_MOTORS:
    m = robot.getDevice(name)
    if m is None:
        sys.exit(f'No encuentro el motor "{name}".')
    m.setVelocity(min(MOTOR_VELOCITY_LIMIT, m.getMaxVelocity()))
    m.getPositionSensor().enable(time_step)
    home.append(m)
robot.step(time_step)
start_angles = [m.getPositionSensor().getValue() for m in home]
print('\nAngulos al arrancar (rad, A a F):', np.round(start_angles, 3))
if max(abs(a) for a in start_angles) > 0.02:
    print('  -> el brazo NO arranca en la pose cero (el mundo se guardo con el brazo en otra pose). Lo llevo a 0.')
for m in home:
    m.setPosition(0.0)
for _ in range(int(15000 / time_step)):
    if not step_ok():
        sys.exit(0)
    if car_contact_detected():
        for motor in home:
            motor.setPosition(motor.getPositionSensor().getValue())
        sys.exit('PARADA DE SEGURIDAD: contacto entre el robot y el auto al buscar la pose cero.')
    if max(abs(m.getPositionSensor().getValue()) for m in home) < 0.005:
        break
print('Angulos en la pose cero (rad, A a F):', np.round([m.getPositionSensor().getValue() for m in home], 3))

# ---------------- Cadena cinematica ----------------
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as f:
    urdf_path = f.name
    f.write(robot.getUrdf().encode('utf-8'))

here = os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(urdf_path, os.path.join(here, 'irb4600.urdf'))  # para reach_study.py
print('URDF guardado en', os.path.join(here, 'irb4600.urdf'))

chain, mask = make_chain(urdf_path)
collision_boxes = load_collision_boxes(urdf_path, chain)

motors, motor_idx = [], []
for i, link in enumerate(chain.links):
    if mask[i]:
        motor = robot.getDevice(link.name)
        motor.setVelocity(min(MOTOR_VELOCITY_LIMIT, motor.getMaxVelocity()))
        motor.getPositionSensor().enable(time_step)
        motors.append(motor)
        motor_idx.append(i)


wait(1.0)  # que el auto termine de asentarse sobre sus amortiguadores antes de medirlo


def current_joints():
    q = [0.0] * len(chain.links)
    for motor, i in zip(motors, motor_idx):
        q[i] = motor.getPositionSensor().getValue()
    return q


def clearance_at(q, base_pos, base_R):
    key = _clearance_cache_key(q, base_pos, base_R)
    cached = clearance_cache.get(key)
    if cached is not None:
        return cached
    result = clearance_to_vehicle(
        chain,
        q,
        collision_boxes,
        base_pos,
        base_R,
        car_center,
        car_R,
        BMW_X5_BODY_BOXES,
        margin_m=COLLISION_MARGIN_M,
        tool_radius_m=PAINT_TOOL_RADIUS_M,
    )
    clearance_cache[key] = result
    return result


def path_has_clearance(target_q, base_pos, base_R):
    return path_has_clearance_from(current_joints(), target_q, base_pos, base_R)


def path_has_clearance_from(start_q, target_q, base_pos, base_R):
    start = np.asarray(start_q, dtype=float)
    target = np.asarray(target_q, dtype=float)
    max_delta = float(np.max(np.abs(target - start)))
    steps = max(1, int(np.ceil(max_delta / COLLISION_SAMPLE_STEP_RAD)))
    minimum_clearance = np.inf
    closest_part = 'robot'
    for index in range(1, steps + 1):
        sample = start + (target - start) * (index / steps)
        safe, clearance, part = clearance_at(sample, base_pos, base_R)
        if clearance < minimum_clearance:
            minimum_clearance, closest_part = clearance, part
        if not safe:
            return False, minimum_clearance, closest_part
    return True, float(minimum_clearance), closest_part


def check_model(label):
    """Compara donde cree el modelo que esta la brida contra donde esta de verdad en Webots."""
    base_p = np.array(arm.getPosition())
    base_R_ = np.array(arm.getOrientation()).reshape(3, 3)
    fk = chain.forward_kinematics(current_joints())
    flange_model = base_p + base_R_ @ (fk[:3, 3] - fk[:3, :3] @ TOOL_TIP)
    d_mm = float(np.linalg.norm(np.array(gun.getPosition()) - flange_model) * 1000)
    c = np.dot(base_R_ @ fk[:3, 2], np.array(gun.getOrientation()).reshape(3, 3)[:, 2])
    ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    print(f'CHEQUEO modelo vs Webots ({label}): brida a {d_mm:.1f} mm, eje de la pistola a {ang:.1f} grados')
    return d_mm, ang


d_chk, a_chk = check_model('pose cero')
if d_chk > 30.0 or a_chk > 3.0:
    print('  *** El modelo NO coincide con Webots. No sigas hasta entender por que (mandame esta consola). ***')
    if not DRY_RUN:
        sys.exit('Corte por seguridad: modelo y robot no coinciden.')


def add_node(text):
    robot.getRoot().getField('children').importMFNodeFromString(-1, text)


def add_polyline(points, color):
    """Linea que une los puntos (un solo nodo, barato aunque tenga cientos de puntos)."""
    if len(points) < 2:
        return
    pts = ', '.join(f'{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}' for p in points)
    idx = ' '.join(str(i) for i in range(len(points))) + ' -1'
    add_node(
        'Shape { '
        f'appearance Appearance {{ material Material {{ diffuseColor 0 0 0 emissiveColor {color[0]} {color[1]} {color[2]} }} }} '
        f'geometry IndexedLineSet {{ coord Coordinate {{ point [ {pts} ] }} coordIndex [ {idx} ] }} '
        '}'
    )


def add_box_wireframe(center, rotation, half_extents, color):
    """Draw a collision-envelope box as one set of disconnected line segments."""
    corners = np.array(
        [
            center + rotation @ (half_extents * np.array([x, y, z]))
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    points = ', '.join(f'{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}' for p in corners)
    indices = ' '.join(f'{a} {b} -1' for a, b in edges)
    add_node(
        'Shape { '
        f'appearance Appearance {{ material Material {{ diffuseColor 0 0 0 '
        f'emissiveColor {color[0]} {color[1]} {color[2]} }} }} '
        f'geometry IndexedLineSet {{ coord Coordinate {{ point [ {points} ] }} '
        f'coordIndex [ {indices} ] }} '
        '}'
    )


def add_sphere(pos, color, radius):
    add_node(
        f'Solid {{ translation {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} '
        f'children [ Shape {{ appearance PBRAppearance {{ baseColor {color[0]} {color[1]} {color[2]} '
        f'roughness 1 metalness 0 }} geometry Sphere {{ radius {radius} }} }} ] }}'
    )


def add_paint_mark(surface_point, surface_normal):
    """Draw a non-physical paint patch on the surface after a reached waypoint."""
    cylinder_axis = np.array([0.0, 1.0, 0.0])
    normal = np.array(surface_normal, dtype=float, copy=True)
    normal /= np.linalg.norm(normal)
    rotation_axis = np.cross(cylinder_axis, normal)
    sin_angle = float(np.linalg.norm(rotation_axis))
    cos_angle = float(np.clip(np.dot(cylinder_axis, normal), -1.0, 1.0))
    if sin_angle < 1e-9:
        rotation_axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0 if cos_angle >= 0.0 else np.pi
    else:
        rotation_axis /= sin_angle
        angle = float(np.arctan2(sin_angle, cos_angle))
    center = np.asarray(surface_point, dtype=float) + normal * 0.003
    add_node(
        f'Pose {{ translation {center[0]:.4f} {center[1]:.4f} {center[2]:.4f} '
        f'rotation {rotation_axis[0]:.5f} {rotation_axis[1]:.5f} '
        f'{rotation_axis[2]:.5f} {angle:.5f} children [ '
        f'Shape {{ appearance PBRAppearance {{ '
        f'baseColor {SIMULATED_PAINT_COLOR[0]} {SIMULATED_PAINT_COLOR[1]} '
        f'{SIMULATED_PAINT_COLOR[2]} roughness 0.8 metalness 0 }} '
        f'geometry Cylinder {{ radius {SIMULATED_PAINT_RADIUS_M:.4f} '
        f'height 0.004 }} }} ] }}'
    )


def go_to(q, timeout=WAYPOINT_TIMEOUT_S):
    """Manda los angulos a los motores y espera a que lleguen (o a que pase el tiempo maximo)."""
    base_pos = np.asarray(arm.getPosition(), dtype=float)
    base_R = np.asarray(arm.getOrientation(), dtype=float).reshape(3, 3)
    safe, clearance, part = path_has_clearance(q, base_pos, base_R)
    if not safe:
        print(
            f'    BLOQUEADO por margen de colision: {part}, '
            f'{clearance * 100:.1f} cm de separacion estimada '
            f'(se requieren {COLLISION_MARGIN_M * 100:.0f} cm).'
        )
        return False
    for motor, i in zip(motors, motor_idx):
        motor.setPosition(float(q[i]))
    for _ in range(max(1, int(timeout * 1000 / time_step))):
        if not step_ok():
            sys.exit(0)
        if car_contact_detected():
            for motor, i in zip(motors, motor_idx):
                motor.setPosition(current_joints()[i])
            sys.exit('PARADA DE SEGURIDAD: Webots detecto contacto entre el robot y el auto.')
        now = current_joints()
        if max(abs(now[i] - q[i]) for i in motor_idx) < REACH_TOL_RAD:
            return True
    return False


# ---------------- Poses del mundo ----------------
base_z0 = float(arm.getPosition()[2])
car_R = np.array(car.getOrientation()).reshape(3, 3)
car_center = np.array(car.getPosition()) + car_R @ CAR_ORIGIN_TO_CENTER
print(f'\nAuto: centro en el mundo {np.round(car_center, 3)}')


def station_world(station):
    """Pose de la base del robot en el mundo para una estacion (g, s, lado)."""
    g, s, side = station[:3]
    yaw_offset_deg = station[3] if len(station) > 3 else 0.0
    pos_car, R_in_car = station_pose(g, s, side, yaw_offset_deg)
    pos = car_center + car_R @ pos_car
    pos[2] = base_z0  # el robot siempre apoya en el piso
    return pos, car_R @ R_in_car


def place_robot(pos, R):
    """Estaciona la base del robot en 'pos' con la rotacion 'R' (solo giro sobre z)."""
    print(
        '  AVISO: reposicionamiento de la base por Supervisor en la simulacion; '
        'no representa un traslado fisico del robot.'
    )
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    arm.getField('translation').setSFVec3f([float(pos[0]), float(pos[1]), float(pos[2])])
    arm.getField('rotation').setSFRotation([0.0, 0.0, 1.0, yaw])
    arm.resetPhysics()
    for _ in range(3):
        step_ok()
        if car_contact_detected():
            sys.exit('PARADA DE SEGURIDAD: contacto entre el brazo y el auto al cambiar de estacion.')
    got = np.array(arm.getPosition())
    if np.linalg.norm(got - pos) > 0.05:
        sys.exit(f'No pude mover la base del robot a {np.round(pos, 3)} (quedo en {np.round(got, 3)}).')


# Contornos de las parcelas sobre el auto (para calibrar CAR_ORIGIN_TO_CENTER y las parcelas).
for name in PATCHES_TO_PAINT:
    p = PATCHES[name]
    corners = [p['center'] + su * p['half_u'] * p['u'] + sv * p['half_v'] * p['v']
               for su, sv in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]]
    add_polyline(car_to_world(corners, car_center, car_R), (1, 1, 0))

n_points = {name: len(raster(PATCHES[name], SPACING, STEP)[0]) for name in PATCHES_TO_PAINT}
done = {name: np.zeros(n_points[name], bool) for name in PATCHES_TO_PAINT}
directly_reached = {name: np.zeros(n_points[name], bool) for name in PATCHES_TO_PAINT}
no_safe_ik = {name: np.zeros(n_points[name], bool) for name in PATCHES_TO_PAINT}
point_margin_blocked = {name: np.zeros(n_points[name], bool) for name in PATCHES_TO_PAINT}
path_margin_blocked = {name: np.zeros(n_points[name], bool) for name in PATCHES_TO_PAINT}
approach_blocked = {name: np.zeros(n_points[name], bool) for name in PATCHES_TO_PAINT}
collision_blocked = {
    name: np.zeros(n_points[name], dtype=bool) for name in PATCHES_TO_PAINT
}
surface_world = {
    name: car_to_world(
        raster(PATCHES[name], SPACING, STEP)[0],
        car_center,
        car_R,
    )
    for name in PATCHES_TO_PAINT
}

# ---------------- Estaciones ----------------
print(f'\nModo: {RUN_MODE}')
if DRY_RUN:
    print(f'{len(STATIONS_TO_RUN)} estaciones: {STATIONS}')
else:
    action = (
        f'deposicion visual de pintura azul (radio {SIMULATED_PAINT_RADIUS_M * 100:.1f} cm)'
        if RUN_MODE == 'paint_test'
        else 'sin activar pintura'
    )
    print(
        f'PRUEBA LIMITADA: estaciones {MOTION_TEST_STATION_INDICES}, '
        f'parcela {MOTION_TEST_PATCH}, velocidad maxima {MOTION_TEST_MAX_VELOCITY:.2f} rad/s; '
        f'{action}.'
    )
if DRY_RUN:
    for vehicle_box in BMW_X5_BODY_BOXES:
        add_box_wireframe(
            car_center + car_R @ vehicle_box.center,
            car_R,
            vehicle_box.half_extents,
            (1.0, 0.0, 1.0),
        )
    print('Envolvente de colision aproximada: cajas magenta sobre el auto.')


def segments(todo, brk):
    """Tramos de indices consecutivos donde todo es True, cortados donde brk es True."""
    out, start = [], None
    for k, v in enumerate(todo):
        if v and start is not None and brk[k]:
            out.append((start, k - 1))
            start = None
        if v and start is None:
            start = k
        if not v and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, len(todo) - 1))
    return out


if not DRY_RUN:
    # Guardamos el brazo detras de la base (lejos del auto) antes de cada cambio de estacion.
    base_pos = np.array(arm.getPosition())
    base_R = np.array(arm.getOrientation()).reshape(3, 3)
    stow_q, e, a = solve(chain, (STOW_TIP), STOW_DIR, current_joints(), 30.0, 15.0)
    stow_ok = e <= 30.0 and a <= 15.0
    if not stow_ok:
        print(f'  aviso: la pose de guardado no es alcanzable (error {e:.0f} mm); uso todos los motores en 0')
        stow_q = [0.0] * len(chain.links)

for n_st, station in STATIONS_TO_RUN:
    clearance_cache.clear()
    pos_w, R_w = station_world(station)
    g, s, side = station[:3]
    yaw_offset_deg = station[3] if len(station) > 3 else 0.0
    print(
        f'\n=== ESTACION {n_st}: g={g} s={s:+} lado {side} '
        f'yaw={yaw_offset_deg:+.0f} grados -> base en {np.round(pos_w, 3)} ==='
    )
    color = STATION_COLORS[(n_st - 1) % len(STATION_COLORS)]

    if DRY_RUN:
        add_sphere(pos_w + np.array([0, 0, 0.1]), color, 0.08)
        seed, stow_error, stow_angle = solve(
            chain,
            STOW_TIP,
            STOW_DIR,
            current_joints(),
            30.0,
            15.0,
            candidate_validator=lambda q: clearance_at(q, pos_w, R_w)[0],
        )
        stow_safe, stow_clearance, stow_part = clearance_at(seed, pos_w, R_w)
        if (
            stow_error > 30.0
            or stow_angle > 15.0
            or not stow_safe
        ):
            print(
                f'  estacion omitida en DRY_RUN: pose de guardado insegura o no alcanzable '
                f'(error {stow_error:.1f} mm, {stow_angle:.1f} grados; '
                f'{stow_part}, {stow_clearance * 100:.1f} cm).'
            )
            continue
    else:
        safe, clearance, part = clearance_at(stow_q, pos_w, R_w)
        if not safe:
            print(
                f'  estacion omitida: la pose de guardado invade el margen de 10 cm '
                f'({part}, {clearance * 100:.1f} cm).'
            )
            continue
        if not go_to(stow_q, timeout=12.0):
            print('  estacion omitida: no se pudo llegar de forma segura a la pose de guardado.')
            continue
        place_robot(pos_w, R_w)
        seed = current_joints()
    base_pos = np.array(arm.getPosition()) if not DRY_RUN else pos_w
    base_R = np.array(arm.getOrientation()).reshape(3, 3) if not DRY_RUN else R_w

    for name in PATCHES_TO_PAINT:
        if DRY_RUN:
            print(
                f'  {name:8s}: calculando IK y validacion para '
                f'{len(done[name])} waypoints...',
                flush=True,
            )
        plan = plan_patch(chain, PATCHES[name], car_center, car_R, base_pos, base_R, seed,
                          SPACING, STEP, POS_TOL_MM, ANG_TOL_DEG,
                          candidate_validator=lambda q: clearance_at(
                              q, base_pos, base_R
                          )[0],
                          progress_callback=(
                              lambda current, total: print(
                                  f'            progreso IK: {current}/{total}',
                                  flush=True,
                              )
                              if DRY_RUN else None
                          ))
        no_safe_ik[name] |= (~plan['ok']) & ~done[name]
        todo = plan['ok'] & ~done[name]
        print(f'  {name:8s}: {int(plan["ok"].sum()):3d} alcanzables de {len(todo):3d}; nuevos por pintar: {int(todo.sum()):3d}', flush=True)
        if not todo.any():
            continue
        clearance_ok = np.zeros(len(todo), dtype=bool)
        minimum_clearance = np.inf
        closest_part = 'robot'
        candidate_indices = np.flatnonzero(todo)
        if DRY_RUN:
            print(
                f'            comprobando margen en {len(candidate_indices)} puntos...',
                flush=True,
            )
        for check_index, k in enumerate(candidate_indices, start=1):
            safe, clearance, part = clearance_at(plan['q'][k], base_pos, base_R)
            if clearance < minimum_clearance:
                minimum_clearance, closest_part = clearance, part
            if plan['safe'][k] and safe:
                clearance_ok[k] = True
            else:
                collision_blocked[name][k] = True
                point_margin_blocked[name][k] = True
            if DRY_RUN and (
                check_index == 1
                or check_index % 10 == 0
                or check_index == len(candidate_indices)
            ):
                print(
                    f'            comprobacion de margen: '
                    f'{check_index}/{len(candidate_indices)}',
                    flush=True,
                )
        if DRY_RUN:
            print('            validando entrada y continuidad de tramos...', flush=True)
        stow_q, stow_error, stow_angle = solve(
            chain,
            STOW_TIP,
            STOW_DIR,
            seed,
            30.0,
            15.0,
            candidate_validator=lambda q: clearance_at(q, base_pos, base_R)[0],
        )
        candidate_segments = segments(clearance_ok, plan['brk'])
        for segment_index, (a0, a1) in enumerate(candidate_segments, start=1):
            if DRY_RUN:
                print(
                    f'            validando tramo {segment_index}/'
                    f'{len(candidate_segments)} ({a0}-{a1})...',
                    flush=True,
                )
            approach_tip = plan['tip_b'][a0] - plan['dir_b'][a0] * APPROACH
            approach_q, approach_error, approach_angle = solve(
                chain,
                approach_tip,
                plan['dir_b'][a0],
                stow_q,
                POS_TOL_MM,
                ANG_TOL_DEG,
                candidate_validator=lambda q: clearance_at(q, base_pos, base_R)[0],
            )
            approach_ok = (
                stow_error <= 30.0
                and stow_angle <= 15.0
                and approach_error <= POS_TOL_MM
                and approach_angle <= ANG_TOL_DEG
            )
            if approach_ok:
                approach_ok, clearance, part = path_has_clearance_from(
                    stow_q, approach_q, base_pos, base_R
                )
                if clearance < minimum_clearance:
                    minimum_clearance, closest_part = clearance, part
            if approach_ok:
                approach_ok, clearance, part = path_has_clearance_from(
                    approach_q, plan['q'][a0], base_pos, base_R
                )
                if clearance < minimum_clearance:
                    minimum_clearance, closest_part = clearance, part
            previous_safe_q = np.asarray(plan['q'][a0], dtype=float)
            segment_points = a1 - a0
            for path_index, k in enumerate(range(a0 + 1, a1 + 1), start=1):
                path_safe, clearance, part = path_has_clearance_from(
                    previous_safe_q,
                    plan['q'][k],
                    base_pos,
                    base_R,
                )
                if clearance < minimum_clearance:
                    minimum_clearance, closest_part = clearance, part
                if path_safe:
                    previous_safe_q = np.asarray(plan['q'][k], dtype=float)
                else:
                    clearance_ok[k] = False
                    collision_blocked[name][k] = True
                    path_margin_blocked[name][k] = True
                if DRY_RUN and (
                    path_index == 1
                    or path_index % 5 == 0
                    or path_index == segment_points
                ):
                    print(
                        f'            recorrido del tramo {segment_index}: '
                        f'{path_index}/{segment_points} puntos comprobados',
                        flush=True,
                    )
            if DRY_RUN:
                print(
                    f'            tramo {segment_index}/{len(candidate_segments)} validado.',
                    flush=True,
                )
            if not approach_ok:
                clearance_ok[a0:a1 + 1] = False
                collision_blocked[name][a0:a1 + 1] = True
                approach_blocked[name][a0:a1 + 1] = True
        blocked_count = int(np.count_nonzero(todo & ~clearance_ok))
        if blocked_count:
            print(
                f'            bloqueados por margen de {COLLISION_MARGIN_M * 100:.0f} cm: '
                f'{blocked_count}; distancia minima estimada {minimum_clearance * 100:.1f} cm '
                f'({closest_part})'
            )
        todo &= clearance_ok
        if not todo.any():
            continue
        segs = segments(todo, plan['brk'])
        jumps = [float(np.abs(np.diff(np.array(plan['q'][a0:a1 + 1]), axis=0)).max()) for a0, a1 in segs if a1 > a0]
        print(f'            {len(segs)} tramo(s); salto maximo de articulacion dentro de un tramo: {max(jumps) if jumps else 0:.2f} rad', flush=True)

        if DRY_RUN:
            for a0, a1 in segments(todo, plan['brk']):
                add_polyline(plan['tip_w'][a0:a1 + 1], color)
            done[name] |= todo
            continue

        real_pos_err, real_ang_err = [], []
        for a0, a1 in segments(todo, plan['brk']):
            # Acercamiento: sobre el primer punto del tramo, alejado de la superficie.
            ap = plan['tip_b'][a0] - plan['dir_b'][a0] * APPROACH
            q_ap, e, a = solve(
                chain,
                ap,
                plan['dir_b'][a0],
                current_joints(),
                POS_TOL_MM,
                ANG_TOL_DEG,
                candidate_validator=lambda q: clearance_at(q, base_pos, base_R)[0],
            )
            if e > POS_TOL_MM or a > ANG_TOL_DEG or not go_to(q_ap, timeout=10.0):
                print(f'    tramo omitido: no se pudo alcanzar con seguridad el acercamiento de {name}.')
                continue
            for k in range(a0, a1 + 1):
                reached = go_to(plan['q'][k])
                gun_pos = np.array(gun.getPosition())
                gun_R = np.array(gun.getOrientation()).reshape(3, 3)
                tip_real = gun_pos + gun_R @ TOOL_TIP
                real_pos_err.append(np.linalg.norm(tip_real - plan['tip_w'][k]) * 1000)
                c = np.dot(gun_R[:, 2], plan['dir_w'][k])
                real_ang_err.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
                if reached:
                    if RUN_MODE == 'paint_test':
                        add_paint_mark(plan['S_w'][k], -plan['dir_w'][k])
                        directly_reached[name][k] = True
                        done[name] |= paint_coverage_mask(
                            surface_world[name],
                            [plan['S_w'][k]],
                            SIMULATED_PAINT_RADIUS_M,
                        )
                    else:
                        add_sphere(plan['S_w'][k], (0.2, 0.9, 0.3), 0.035)
                        directly_reached[name][k] = True
                        done[name][k] = True
                else:
                    print(f'    aviso: punto {k} de {name} omitido por tiempo o margen de seguridad')
            # Alejamiento: sobre el ultimo punto del tramo.
            rt = plan['tip_b'][a1] - plan['dir_b'][a1] * APPROACH
            q_rt, e, a = solve(
                chain,
                rt,
                plan['dir_b'][a1],
                current_joints(),
                POS_TOL_MM,
                ANG_TOL_DEG,
                candidate_validator=lambda q: clearance_at(q, base_pos, base_R)[0],
            )
            if e <= POS_TOL_MM and a <= ANG_TOL_DEG:
                go_to(q_rt, timeout=10.0)
        if real_pos_err:
            print(f'    punta real vs planificada: media {np.mean(real_pos_err):6.1f} mm, max {np.max(real_pos_err):6.1f} mm'
                  f' | orientacion: media {np.mean(real_ang_err):4.1f}, max {np.max(real_ang_err):4.1f} grados')
        seed = current_joints()
        if RUN_MODE == 'paint_test':
            remaining = int(np.count_nonzero(~done[name]))
            print(
                f'    clusters restantes para otras estaciones: {remaining} waypoint(s).',
                flush=True,
            )

# ---------------- Resumen ----------------
coverage_label = (
    'COBERTURA PLANIFICADA'
    if DRY_RUN
    else (
        'WAYPOINTS CUBIERTOS POR HUELLA DE PINTURA SIMULADA'
        if RUN_MODE == 'paint_test'
        else 'WAYPOINTS ALCANZADOS EN PRUEBA DE MOVIMIENTO'
    )
)
print(f'\n{coverage_label} (waypoints del zigzag, no area fisica):')
for name in PATCHES_TO_PAINT:
    painted = int(done[name].sum())
    total = len(done[name])
    print(
        f'  {name:8s}: {painted:3d} de {total:3d} waypoints '
        f'({100 * done[name].mean():5.1f}%); '
        f'{int(np.count_nonzero(collision_blocked[name] & ~done[name]))} pendientes por margen'
    )
    if RUN_MODE == 'paint_test':
        pending = ~done[name]
        print(
            f'            alcanzados directamente: {int(directly_reached[name].sum())}; '
            f'pendientes: {int(pending.sum())} '
            f'motivos que pueden solaparse entre estaciones '
            f'(sin IK segura - alcance o margen: '
            f'{int(np.count_nonzero(pending & no_safe_ik[name]))}, '
            f'margen en punto: {int(np.count_nonzero(pending & point_margin_blocked[name]))}, '
            f'margen en recorrido: {int(np.count_nonzero(pending & path_margin_blocked[name]))}, '
            f'aproximacion insegura: {int(np.count_nonzero(pending & approach_blocked[name]))}).'
        )
    for index, point in enumerate(surface_world[name]):
        if not done[name][index]:
            add_sphere(point, (1.0, 0.15, 0.1), 0.025)
        elif DRY_RUN:
            add_sphere(point, (0.1, 0.8, 1.0), 0.025)

if DRY_RUN:
    print(
        '\nDRY_RUN = True: los marcadores celestes son cobertura planificada segura; '
        'los rojos son puntos no cubiertos o bloqueados. No se ejecuto pintura.'
    )
else:
    if RUN_MODE == 'paint_test':
        print('\nMarcas azules: deposicion visual simulada en waypoints alcanzados.')
        print('No se modifico la textura del auto ni se activo pintura real.')
    else:
        print('\nMarcadores verdes: waypoints alcanzados; rojos: pendientes o bloqueados.')
        print('No se activo pintura. El porcentaje mide waypoints, no area pintada.')
while step_ok():
    pass