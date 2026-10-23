"""Conservative robot-to-vehicle clearance checks using URDF collision boxes."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class CollisionBox:
    link_index: int
    link_name: str
    offset: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray


@dataclass(frozen=True)
class VehicleBox:
    name: str
    center: np.ndarray
    half_extents: np.ndarray


# Approximate BmwX5 body volumes based on the assessment's 4.8 x 1.9 x 1.6 m
# reference dimensions and the top RangeFinder profile; these are not exact mesh colliders.
BMW_X5_BODY_BOXES = (
    VehicleBox('lower_body', np.array([0.0, 0.0, 0.55]), np.array([2.05, 0.90, 0.30])),
    VehicleBox('hood_rear', np.array([1.25, 0.0, 0.821]), np.array([0.30, 0.82, 0.074])),
    VehicleBox('hood', np.array([1.75, 0.0, 0.749]), np.array([0.30, 0.82, 0.074])),
    VehicleBox('hood_front', np.array([2.25, 0.0, 0.676]), np.array([0.30, 0.82, 0.074])),
    VehicleBox('cabin_shell', np.array([-0.78, 0.0, 1.08]), np.array([1.10, 0.70, 0.25])),
    VehicleBox('cabin_roof', np.array([-0.78, 0.0, 1.38]), np.array([1.10, 0.65, 0.06])),
    VehicleBox('rear', np.array([-1.75, 0.0, 0.88]), np.array([0.55, 0.82, 0.24])),
    VehicleBox('front_bumper', np.array([2.25, 0.0, 0.55]), np.array([0.15, 0.88, 0.24])),
    VehicleBox('rear_bumper', np.array([-2.25, 0.0, 0.55]), np.array([0.15, 0.88, 0.24])),
    VehicleBox('front_left_wheel', np.array([1.5, 0.82, 0.38]), np.array([0.42, 0.16, 0.36])),
    VehicleBox('front_right_wheel', np.array([1.5, -0.82, 0.38]), np.array([0.42, 0.16, 0.36])),
    VehicleBox('rear_left_wheel', np.array([-1.5, 0.82, 0.38]), np.array([0.42, 0.16, 0.36])),
    VehicleBox('rear_right_wheel', np.array([-1.5, -0.82, 0.38]), np.array([0.42, 0.16, 0.36])),
)


def _rotation_from_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def load_collision_boxes(urdf_path, chain):
    """Load supported URDF box colliders and map them to IKPy link-frame indices."""
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib['name']: link for link in root.findall('link')}
    joint_children = {
        joint.attrib['name']: joint.find('child').attrib['link']
        for joint in root.findall('joint')
    }
    child_links = set(joint_children.values())
    root_links = set(links) - child_links
    if len(root_links) != 1:
        raise ValueError('Expected exactly one root link in the robot URDF.')

    chain_link_names = {0: next(iter(root_links))}
    for index, chain_link in enumerate(chain.links[1:], start=1):
        child_link = joint_children.get(chain_link.name)
        if child_link is not None:
            chain_link_names[index] = child_link
        elif chain_link.name in links:
            chain_link_names[index] = chain_link.name

    collision_link_names = {
        link.attrib['name'] for link in links.values() if link.findall('collision')
    }
    unmapped_links = collision_link_names - set(chain_link_names.values())
    if unmapped_links:
        raise ValueError(
            'Collision geometry is not represented in the IKPy chain: '
            + ', '.join(sorted(unmapped_links))
        )

    boxes = []
    for index, link_name in chain_link_names.items():
        link = links[link_name]
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            box = None if geometry is None else geometry.find('box')
            if box is None:
                raise ValueError(
                    f'Unsupported collision geometry on URDF link {link_name}; '
                    'only box colliders are supported.'
                )
            size = np.fromstring(box.attrib['size'], sep=' ')
            if size.shape != (3,) or np.any(size <= 0.0):
                raise ValueError(f'Invalid box dimensions on URDF link {link_name}.')
            origin = collision.find('origin')
            offset = np.zeros(3) if origin is None else np.fromstring(
                origin.attrib.get('xyz', '0 0 0'), sep=' '
            )
            rpy = np.zeros(3) if origin is None else np.fromstring(
                origin.attrib.get('rpy', '0 0 0'), sep=' '
            )
            boxes.append(
                CollisionBox(
                    link_index=index,
                    link_name=link_name,
                    offset=offset,
                    rotation=_rotation_from_rpy(rpy),
                    half_extents=size / 2.0,
                )
            )

    if not boxes:
        raise ValueError('The robot URDF contains no supported collision boxes.')
    return boxes


def _box_clearance_lower_bound(center, rotation, half_extents, vehicle_half_extents):
    """Return a conservative separating-axis lower bound for two oriented boxes."""
    vehicle_axes = np.eye(3)
    axes = [vehicle_axes[:, index] for index in range(3)]
    axes.extend(rotation[:, index] for index in range(3))
    axes.extend(
        np.cross(vehicle_axes[:, i], rotation[:, j])
        for i in range(3)
        for j in range(3)
    )

    maximum_gap = 0.0
    for axis in axes:
        magnitude = np.linalg.norm(axis)
        if magnitude < 1e-9:
            continue
        unit_axis = axis / magnitude
        vehicle_radius = np.dot(vehicle_half_extents, np.abs(unit_axis))
        box_radius = np.dot(half_extents, np.abs(rotation.T @ unit_axis))
        gap = abs(np.dot(center, unit_axis)) - vehicle_radius - box_radius
        maximum_gap = max(maximum_gap, float(gap))
    return max(0.0, maximum_gap)


def clearance_to_vehicle(
    chain,
    joints,
    collision_boxes,
    base_position,
    base_rotation,
    vehicle_center,
    vehicle_rotation,
    vehicle_boxes,
    margin_m=0.10,
    tool_radius_m=0.035,
):
    """Check every robot collider and the nozzle against a union of vehicle boxes.

    A returned clearance below ``margin_m`` is treated as unsafe. The separating-axis
    distance is a conservative lower bound, so some near-corner poses can be rejected.
    """
    if margin_m < 0.0 or tool_radius_m < 0.0:
        raise ValueError('Collision margin and tool radius must be non-negative.')
    if not vehicle_boxes:
        raise ValueError('At least one vehicle collision box is required.')
    fk = chain.forward_kinematics(joints, full_kinematics=True)
    minimum_clearance = np.inf
    closest_part = 'robot'

    for box in collision_boxes:
        link_frame = fk[box.link_index]
        world_rotation = base_rotation @ link_frame[:3, :3] @ box.rotation
        world_center = (
            base_position
            + base_rotation @ link_frame[:3, 3]
            + base_rotation @ link_frame[:3, :3] @ box.offset
        )
        center_in_car = vehicle_rotation.T @ (world_center - vehicle_center)
        rotation_in_car = vehicle_rotation.T @ world_rotation
        for vehicle_box in vehicle_boxes:
            clearance = _box_clearance_lower_bound(
                center_in_car - vehicle_box.center,
                rotation_in_car,
                box.half_extents,
                vehicle_box.half_extents,
            )
            if clearance < minimum_clearance:
                minimum_clearance = clearance
                closest_part = f'{box.link_name}/{vehicle_box.name}'

    tool_frame = fk[-1]
    tool_world = base_position + base_rotation @ tool_frame[:3, 3]
    tool_in_car = vehicle_rotation.T @ (tool_world - vehicle_center)
    for vehicle_box in vehicle_boxes:
        outside = np.maximum(
            np.abs(tool_in_car - vehicle_box.center) - vehicle_box.half_extents,
            0.0,
        )
        tool_clearance = max(0.0, float(np.linalg.norm(outside)) - tool_radius_m)
        if tool_clearance < minimum_clearance:
            minimum_clearance = tool_clearance
            closest_part = f'punta de pintura/{vehicle_box.name}'

    return minimum_clearance >= margin_m, float(minimum_clearance), closest_part
