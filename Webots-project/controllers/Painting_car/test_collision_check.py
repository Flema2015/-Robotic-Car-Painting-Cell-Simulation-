import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from collision_check import (
    BMW_X5_BODY_BOXES,
    CollisionBox,
    VehicleBox,
    _box_clearance_lower_bound,
    clearance_to_vehicle,
    load_collision_boxes,
)


class CollisionCheckTests(unittest.TestCase):
    def test_vehicle_boxes_cover_rgbd_calibrated_roof_and_hood_samples(self):
        samples = {
            'cabin_shell': (
                np.array([-1.8, 0.0, 1.10]),
                np.array([0.3, 0.0, 1.10]),
            ),
            'cabin_roof': (
                np.array([-1.8, 0.0, 1.38]),
                np.array([0.3, 0.0, 1.38]),
            ),
            'hood': (
                np.array([1.1, 0.0, 0.86]),
                np.array([2.2, 0.0, 0.70]),
            ),
        }
        boxes = {box.name: box for box in BMW_X5_BODY_BOXES}
        hood_boxes = tuple(box for box in BMW_X5_BODY_BOXES if box.name.startswith('hood'))

        for name, points in samples.items():
            for point in points:
                containing_boxes = hood_boxes if name == 'hood' else (boxes[name],)
                self.assertTrue(any(
                    np.all(np.abs(point - box.center) <= box.half_extents)
                    for box in containing_boxes
                ), f'{name} box does not contain calibrated point {point}')

    def test_axis_aligned_boxes_report_their_gap(self):
        clearance = _box_clearance_lower_bound(
            center=np.array([1.5, 0.0, 0.0]),
            rotation=np.eye(3),
            half_extents=np.array([0.3, 0.3, 0.3]),
            vehicle_half_extents=np.array([1.0, 1.0, 1.0]),
        )
        self.assertAlmostEqual(clearance, 0.2)

    def test_overlapping_boxes_have_zero_clearance(self):
        clearance = _box_clearance_lower_bound(
            center=np.array([1.2, 0.0, 0.0]),
            rotation=np.eye(3),
            half_extents=np.array([0.3, 0.3, 0.3]),
            vehicle_half_extents=np.array([1.0, 1.0, 1.0]),
        )
        self.assertEqual(clearance, 0.0)

    def test_separated_rotated_boxes_have_a_positive_lower_bound(self):
        angle = np.pi / 4.0
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        clearance = _box_clearance_lower_bound(
            center=np.array([1.8, 0.0, 0.0]),
            rotation=rotation,
            half_extents=np.array([0.3, 0.3, 0.3]),
            vehicle_half_extents=np.array([1.0, 1.0, 1.0]),
        )
        self.assertGreater(clearance, 0.1)

    def test_urdf_collision_boxes_map_to_ikpy_link_indices(self):
        urdf = """<robot name="test">
          <link name="base"/>
          <link name="arm">
            <collision><origin xyz="0 0 1"/><geometry><box size="2 4 6"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base"/><child link="arm"/>
          </joint>
        </robot>"""
        chain = SimpleNamespace(
            links=[
                SimpleNamespace(name='Base link'),
                SimpleNamespace(name='shoulder'),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'robot.urdf'
            path.write_text(urdf, encoding='utf-8')
            boxes = load_collision_boxes(path, chain)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].link_index, 1)
        self.assertEqual(boxes[0].link_name, 'arm')
        np.testing.assert_array_equal(boxes[0].offset, [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(boxes[0].half_extents, [1.0, 2.0, 3.0])

    def test_clearance_enforces_ten_centimeters(self):
        class FakeChain:
            def forward_kinematics(self, joints, full_kinematics=False):
                box_frame = np.eye(4)
                box_frame[0, 3] = 1.29
                tool_frame = np.eye(4)
                tool_frame[0, 3] = 4.0
                return [box_frame, tool_frame]

        boxes = [
            CollisionBox(
                link_index=0,
                link_name='arm',
                offset=np.zeros(3),
                rotation=np.eye(3),
                half_extents=np.array([0.2, 0.2, 0.2]),
            )
        ]
        safe, clearance, part = clearance_to_vehicle(
            FakeChain(),
            [],
            boxes,
            np.zeros(3),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            (VehicleBox('body', np.zeros(3), np.ones(3)),),
            margin_m=0.10,
            tool_radius_m=0.035,
        )

        self.assertFalse(safe)
        self.assertAlmostEqual(clearance, 0.09)
        self.assertEqual(part, 'arm/body')

    def test_nozzle_above_hood_is_not_inside_a_full_vehicle_box(self):
        class FakeChain:
            def forward_kinematics(self, joints, full_kinematics=False):
                tool_frame = np.eye(4)
                tool_frame[:3, 3] = [1.5, 0.0, 1.25]
                return [tool_frame]

        safe, clearance, part = clearance_to_vehicle(
            FakeChain(),
            [],
            [],
            np.zeros(3),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            BMW_X5_BODY_BOXES,
            margin_m=0.10,
            tool_radius_m=0.035,
        )

        self.assertTrue(safe)
        self.assertGreater(clearance, 0.10)
        self.assertEqual(part, 'punta de pintura/hood_rear')


if __name__ == '__main__':
    unittest.main()
