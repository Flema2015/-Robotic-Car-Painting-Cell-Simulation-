import sys
import tempfile
import warnings
import numpy as np

try:
    from ikpy.chain import Chain 
except ImportError:
    sys.exit('there is no module named "ikpy". Install it with "pip install ikpy"')

from controller import Supervisor

# ---------------- Parameters ----------------
# the robot's goal based on the robot's local coordinate system.
# x = foward, y = To the Left, z = To the top.
TARGET_LOCAL = np.array([1.5, 0.0, 1.0])
SETTLE_TIME_S = 8.0  # constant created to allow the robot to reach the target position before checking the error.

# ---------------- Webots ----------------
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
arm = robot.getSelf()

# To ensure that the robot supervisor is in TRUE mode, otherwise the robot will not be able to move.
if arm is None:
    sys.exit('This robot is NOT a Supervisor. Set the supervisor to TRUE on the robot, '
             'save the world (Ctrl+S) and reset the simulation (Ctrl+Shift+T).')

    
if not np.isfinite(arm.getPosition()).all():
    sys.exit("The robot's supervisor has no valid position. Set the supervisor to TRUE on the robot, "
             'save the world (Ctrl+S) and reset the simulation (Ctrl+Shift+T).')
# ----------------  Kinematics (based on ikpy) ----------------
# Webots generates a URDF description of the robot; ikpy converts it into a mathematical model.
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
#retorna La lista de posiciones de cada articulación según el objetivo. 
ik = chain.inverse_kinematics(TARGET_LOCAL, initial_position=current_joints()) 
for motor, i in zip(motors, motor_idx):
    motor.setPosition(ik[i])
    
    # print(f"Posicion de motor {motor.getName()}: {ik[i]}")
# Esperamos a que el brazo llegue.
for _ in range(int(SETTLE_TIME_S * 1000 / time_step)):
    if robot.step(time_step) == -1:
        sys.exit(0)

# ---------------- Verificacion ----------------
# Cinematica directa: con los angulos REALES de los sensores, donde quedo la brida?
reached = chain.forward_kinematics(current_joints())[:3, 3]
error_mm = np.linalg.norm(reached - TARGET_LOCAL) * 1000
print(f'La brida llego a {np.round(reached, 3)} | error = {error_mm:.1f} mm')

# El controlador tiene que seguir vivo: si termina, los motores quedan sin orden
# y el brazo se cae por la gravedad.
while robot.step(time_step) != -1:
    pass