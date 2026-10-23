import unittest

import numpy as np

from cell_model import (
    PATCHES,
    STATIONS,
    paint_coverage_mask,
    plan_patch,
    solve,
    station_pose,
)


class FakeLink:
    bounds = None


class TwoBranchChain:
    def __init__(self):
        self.links = [FakeLink() for _ in range(6)]
        self.active_links_mask = [False] * 6
        self.target = None

    def inverse_kinematics(self, tip, target_orientation, orientation_mode, initial_position):
        self.target = np.asarray(tip, dtype=float)
        joints = np.zeros(6)
        joints[0] = 1.0 if initial_position is not None else 2.0
        return joints

    def forward_kinematics(self, joints):
        frame = np.eye(4)
        frame[:3, 3] = self.target
        return frame


class CollisionAwareSolveTests(unittest.TestCase):
    def test_fourth_station_is_the_complementary_roof_candidate(self):
        self.assertEqual(STATIONS[-1], (1.5, -0.5, 'R', -30.0))

    def test_roof_patch_matches_measured_rgbd_height_and_extent(self):
        roof = PATCHES['techo']

        self.assertAlmostEqual(roof['center'][2], 1.38)
        self.assertAlmostEqual(roof['center'][0] - roof['half_u'], -1.88)
        self.assertAlmostEqual(roof['center'][0] + roof['half_u'], 0.32)
        self.assertAlmostEqual(roof['half_v'], 0.65)

    def test_hood_patch_follows_measured_downward_slope(self):
        hood = PATCHES['capo']
        tangent = hood['u']
        normal = hood['normal']

        self.assertAlmostEqual(tangent[2] / tangent[0], -0.1466, places=3)
        self.assertAlmostEqual(np.dot(tangent, normal), 0.0, places=4)
        self.assertGreater(normal[0], 0.0)

    def test_validator_prefers_safe_zero_seed_solution_to_nearest_unsafe_one(self):
        chain = TwoBranchChain()
        seed = np.zeros(6)

        joints, position_error, angle_error = solve(
            chain,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0]),
            seed,
            candidate_validator=lambda candidate: candidate[0] == 2.0,
        )

        self.assertEqual(joints[0], 2.0)
        self.assertAlmostEqual(position_error, 0.0)
        self.assertAlmostEqual(angle_error, 0.0)

    def test_without_validator_solver_keeps_existing_nearest_solution(self):
        chain = TwoBranchChain()

        joints, _, _ = solve(
            chain,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0]),
            np.zeros(6),
        )

        self.assertEqual(joints[0], 1.0)

    def test_station_yaw_offset_rotates_in_place_without_changing_position(self):
        position, base_rotation = station_pose(1.5, 0.5, 'L')
        rotated_position, rotated = station_pose(1.5, 0.5, 'L', 30.0)

        np.testing.assert_array_equal(rotated_position, position)
        np.testing.assert_allclose(rotated.T @ rotated, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotated), 1.0)
        self.assertFalse(np.allclose(rotated, base_rotation))

    def test_plan_patch_reports_progress_at_start_and_completion(self):
        patch = {
            'center': np.zeros(3),
            'u': np.array([1.0, 0.0, 0.0]),
            'v': np.array([0.0, 1.0, 0.0]),
            'normal': np.array([0.0, 0.0, 1.0]),
            'half_u': 0.05,
            'half_v': 0.05,
        }
        progress = []

        plan_patch(
            TwoBranchChain(),
            patch,
            np.zeros(3),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.zeros(6),
            progress_callback=lambda current, total: progress.append((current, total)),
        )

        self.assertEqual(progress, [(1, 4), (4, 4)])

    def test_paint_coverage_uses_simulated_footprint_radius(self):
        surface = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.075, 0.0, 0.0],
                [0.076, 0.0, 0.0],
                [0.0, 0.0, 0.08],
            ]
        )
        painted = np.array([[0.0, 0.0, 0.0]])

        np.testing.assert_array_equal(
            paint_coverage_mask(surface, painted, 0.075),
            [True, True, False, False],
        )

    def test_paint_coverage_handles_no_painted_points(self):
        surface = np.zeros((2, 3))

        np.testing.assert_array_equal(
            paint_coverage_mask(surface, [], 0.075),
            [False, False],
        )

    def test_front_and_rear_stations_face_toward_vehicle_center(self):
        front_position, front_rotation = station_pose(1.5, 0.25, 'F')
        rear_position, rear_rotation = station_pose(1.5, 0.25, 'B')

        np.testing.assert_array_equal(front_position, [1.5, 0.25, 0.0])
        np.testing.assert_array_equal(rear_position, [-1.5, 0.25, 0.0])
        np.testing.assert_array_equal(front_rotation[:, 0], [-1.0, 0.0, 0.0])
        np.testing.assert_array_equal(rear_rotation[:, 0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.det(front_rotation), 1.0)
        self.assertAlmostEqual(np.linalg.det(rear_rotation), 1.0)


if __name__ == '__main__':
    unittest.main()
