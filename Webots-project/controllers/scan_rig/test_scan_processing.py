import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scan_processing import (
    add_neighbor_supported_candidates,
    bgra_to_rgb,
    exclude_image_regions,
    exclude_rgb_range,
    floor_depth_limit_map,
    make_red_body_mask,
    retain_polygon_region,
    save_png,
)


class ScanProcessingTests(unittest.TestCase):
    def test_neighbor_expansion_adds_only_adjacent_candidates(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        candidates = np.zeros((5, 5), dtype=bool)
        candidates[1, 1] = True
        candidates[2, 3] = True
        candidates[4, 4] = True
        expanded = add_neighbor_supported_candidates(mask, candidates)

        self.assertTrue(expanded[1, 1])
        self.assertTrue(expanded[2, 3])
        self.assertFalse(expanded[4, 4])
        self.assertEqual(int(expanded.sum()), 3)
        self.assertEqual(int(mask.sum()), 1)

    def test_neighbor_expansion_does_not_propagate_in_a_chain(self):
        mask = np.zeros((1, 4), dtype=bool)
        mask[0, 0] = True
        candidates = np.ones((1, 4), dtype=bool)

        expanded = add_neighbor_supported_candidates(mask, candidates)

        np.testing.assert_array_equal(expanded, [[True, True, False, False]])

    def test_neighbor_expansion_requires_matching_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'matching height and width'):
            add_neighbor_supported_candidates(
                np.ones((1, 2), dtype=bool),
                np.ones((2, 1), dtype=bool),
            )

    def test_bgra_conversion_restores_rgb_channel_order(self):
        bgra = np.array([[[10, 20, 230, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(bgra_to_rgb(bgra), [[[230, 20, 10]]])

    def test_red_mask_excludes_grey_glass_and_red_floor(self):
        rgb = np.array(
            [[[180, 30, 35], [70, 72, 75], [180, 30, 35], [180, 30, 35]]],
            dtype=np.uint8,
        )
        depth = np.array([[4.8, 4.7, 5.9, np.inf]], dtype=np.float32)
        mask = make_red_body_mask(rgb, depth, floor_range_m=5.95)
        np.testing.assert_array_equal(mask, [[True, False, False, False]])

    def test_rgb_range_excludes_arm_color_and_preserves_car_red(self):
        rgb = np.array(
            [[[72, 42, 36], [180, 30, 35], [70, 72, 75]]],
            dtype=np.uint8,
        )
        mask = np.array([[True, True, False]])
        filtered = exclude_rgb_range(mask, rgb, (50, 30, 20), (85, 60, 60))
        np.testing.assert_array_equal(filtered, [[False, True, False]])

    def test_rgb_range_requires_matching_image_dimensions(self):
        mask = np.ones((1, 2), dtype=bool)
        rgb = np.zeros((2, 1, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'matching height and width'):
            exclude_rgb_range(mask, rgb, (50, 30, 20), (85, 60, 60))

    def test_rgb_range_rejects_invalid_bounds(self):
        mask = np.ones((1, 1), dtype=bool)
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'ordered values'):
            exclude_rgb_range(mask, rgb, (90, 30, 20), (85, 60, 60))

    def test_image_region_filter_clears_only_the_configured_rectangle(self):
        mask = np.ones((4, 5), dtype=bool)
        filtered = exclude_image_regions(mask, ((1, 1, 4, 3),))
        self.assertEqual(int(filtered.sum()), 14)
        self.assertFalse(filtered[1:3, 1:4].any())
        self.assertTrue(filtered[0, 0])
        self.assertTrue(mask.all())

    def test_image_region_filter_rejects_out_of_bounds_rectangles(self):
        with self.assertRaisesRegex(ValueError, 'within the image'):
            exclude_image_regions(np.ones((2, 2)), ((0, 0, 3, 1),))

    def test_polygon_region_keeps_interior_and_boundary_pixels(self):
        mask = np.ones((5, 5), dtype=bool)
        filtered = retain_polygon_region(mask, ((1, 1), (3, 1), (3, 3), (1, 3)))
        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        np.testing.assert_array_equal(filtered, expected)
        self.assertTrue(mask.all())

    def test_polygon_region_rejects_invalid_vertices(self):
        with self.assertRaisesRegex(ValueError, 'at least three'):
            retain_polygon_region(np.ones((3, 3)), ((0, 0), (1, 1)))

    def test_mask_requires_aligned_camera_and_depth_dimensions(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.zeros((3, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'matching height and width'):
            make_red_body_mask(rgb, depth, floor_range_m=5.95)

    def test_top_view_floor_limit_is_constant_for_parallel_rays(self):
        limits = floor_depth_limit_map(
            320, 240, 1.309, (-0.002, -0.395, 6.0), (0.0, 1.0, 0.0, 1.5708)
        )
        self.assertAlmostEqual(float(limits[120, 160]), 5.88, places=2)
        self.assertLess(float(np.max(limits)), 5.89)

    def test_side_view_floor_limit_changes_with_image_row(self):
        limits = floor_depth_limit_map(
            320, 240, 1.745329, (3.0, -0.395, 1.5),
            (-0.090476903, 0.0, 0.995898554, 3.141592654)
        )
        self.assertTrue(np.isinf(limits[0]).any())
        self.assertTrue(np.isfinite(limits[-1]).any())
        self.assertGreater(float(np.nanmin(limits)), 0.0)

    def test_png_writer_creates_valid_header_dimensions(self):
        image = np.zeros((4, 7, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'test.png'
            save_png(path, image)
            data = path.read_bytes()

        self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(struct.unpack('>II', data[16:24]), (7, 4))


if __name__ == '__main__':
    unittest.main()
