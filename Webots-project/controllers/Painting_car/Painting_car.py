"""Paso 1: llevar la brida (flange) del ABB IRB 4600/40 a un punto y verificar el resultado.

La brida es la placa de montaje al final del brazo, donde se atornilla la herramienta.
Requisito: el robot debe tener el campo supervisor en TRUE (y el mundo guardado).
"""

import sys
import tempfile
import warnings
import numpy as np

try:
    from ikpy.chain import Chain
except ImportError:
    sys.exit('Falta ikpy. Instalalo con: pip install ikpy')

from controller import Supervisor

# ---------------- Parametros ----------------
# Punto objetivo en el sistema de coordenadas de la BASE del robot, en metros:
# x = hacia adelante, y = hacia la izquierda, z = hacia arriba.
TARGET_LOCAL = np.array([1.5, 0.0, 1.0])
SETTLE_TIME_S = 8.0  # segundos de simulacion que esperamos a que el brazo llegue

# ---------------- Webots ----------------
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
arm = robot.getSelf()

# Si el campo supervisor del robot esta en FALSE, getSelf() igual devuelve un objeto,
# pero las llamadas que hagamos con el devuelven NaN. Probamos una y cortamos aca
# con un mensaje claro en vez de dejar que falle mas adelante.
if not np.all(np.isfinite(np.array(arm.getPosition(), dtype=float))):
    sys.exit('Este robot NO es Supervisor. Pone supervisor TRUE en el robot, '
             'guarda el mundo con el boton de guardar de la barra superior y '
             'recarga el mundo (Reload World).')

# ---------------- Cadena cinematica (ikpy) ----------------
# Webots genera una descripcion URDF del robot; ikpy la convierte en un modelo matematico.
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as f:
    urdf_path = f.name
    f.write(robot.getUrdf().encode('utf-8'))

# 1) Cargamos la cadena solo para leer los nombres de los eslabones.
#    Sin mascara, ikpy avisa que los eslabones fijos estan "activos": es inofensivo,
#    asi que silenciamos esos avisos solo en esta carga.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    probe = Chain.from_urdf_file(urdf_path)
mask = ['motor' in link.name for link in probe.links]
# 2) La cargamos de nuevo indicando que eslabones tienen motor (los unicos que ikpy puede mover).
chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)

print('Eslabones de la cadena:')
for i, link in enumerate(chain.links):
    print(f'  [{i}] {link.name}  (con motor: {mask[i]})')

# ---------------- Motores y sensores ----------------
motors, motor_idx = [], []
for i, link in enumerate(chain.links):
    if mask[i]:
        motor = robot.getDevice(link.name)
        motor.setVelocity(1.0)  # rad/s: lento a proposito para ver el movimiento
        motor.getPositionSensor().enable(time_step)
        motors.append(motor)
        motor_idx.append(i)

robot.step(time_step)  # un paso para que los sensores tengan valores validos


def current_joints():
    """Angulos actuales en el formato que espera ikpy (un valor por eslabon)."""
    q = [0.0] * len(chain.links)
    for motor, i in zip(motors, motor_idx):
        q[i] = motor.getPositionSensor().getValue()
    return q


# ---------------- Coordenadas: base del robot -> mundo ----------------
base_pos = np.array(arm.getPosition())
R = np.array(arm.getOrientation()).reshape(3, 3)  # rotacion local -> mundo
target_world = base_pos + R @ TARGET_LOCAL

# NaN = "not a number": numpy lo arrastra en silencio por todas las cuentas.
# Si aparece, algo salio mal antes, asi que paramos.
if not np.all(np.isfinite(target_world)):
    sys.exit(f'El objetivo en coordenadas del mundo no es valido: {target_world}')

print(f'Objetivo (base del robot): {TARGET_LOCAL}')
print(f'Objetivo (mundo):          {np.round(target_world, 3)}')

# Marcador visual: una esfera celeste en el punto objetivo.
marker = (
    f'Solid {{ translation {target_world[0]} {target_world[1]} {target_world[2]} '
    'children [ Shape { appearance PBRAppearance { baseColor 0 0.6 1 roughness 1 metalness 0 } '
    'geometry Sphere { radius 0.03 } } ] }'
)
robot.getRoot().getField('children').importMFNodeFromString(-1, marker)

# ---------------- Cinematica inversa ----------------
ik = chain.inverse_kinematics(TARGET_LOCAL, initial_position=current_joints())
for motor, i in zip(motors, motor_idx):
    motor.setPosition(ik[i])

# Esperamos a que el brazo llegue.
for _ in range(int(SETTLE_TIME_S * 1000 / time_step)):
    if robot.step(time_step) == -1:
        sys.exit(0)

# ---------------- Verificacion ----------------
# Cinematica directa: con los angulos REALES de los sensores, donde quedo la brida?
reached = chain.forward_kinematics(current_joints())[:3, 3]
error_mm = np.linalg.norm(reached - TARGET_LOCAL) * 1000
print(f'La brida llego a {np.round(reached, 3)} | error = {error_mm:.1f} mm')

# ---------------- Comprobacion contra el mundo real ----------------
# Si el Solid de la pistola tiene DEF GUN (se escribe en el panel de abajo del arbol),
# comparamos donde esta DE VERDAD en Webots contra donde calculo ikpy que deberia estar.
gun = robot.getFromDef('GUN')
if gun is None:
    print('(Sin DEF GUN: salteo la comprobacion contra el mundo)')
else:
    gun_pos = np.array(gun.getPosition())
    gun_axes = np.array(gun.getOrientation()).reshape(3, 3)  # columnas = ejes locales en el mundo
    diff_mm = (gun_pos - target_world) * 1000
    print(f'Solid de la pistola en el mundo: {np.round(gun_pos, 3)}')
    print(f'Diferencia contra el objetivo (mm): {np.round(diff_mm, 1)}')
    print(f'Eje z de la pistola en el mundo: {np.round(gun_axes[:, 2], 3)}')

# El controlador tiene que seguir vivo: si termina, los motores quedan sin orden
# y el brazo se cae por la gravedad.
while robot.step(time_step) != -1:
    pass