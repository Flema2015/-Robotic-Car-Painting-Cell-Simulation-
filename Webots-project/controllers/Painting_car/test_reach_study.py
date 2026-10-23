import unittest

import numpy as np

from reach_study import best_combinations, choose_recommendation, coverage


class ReachStudyTests(unittest.TestCase):
    def setUp(self):
        self.stations = [('a',), ('b',), ('c',)]
        self.point_counts = {'capo': 4, 'techo': 2}
        self.flags = {
            ('a',): {
                'capo': np.array([True, True, False, False]),
                'techo': np.array([True, False]),
            },
            ('b',): {
                'capo': np.array([False, False, True, True]),
                'techo': np.array([False, True]),
            },
            ('c',): {
                'capo': np.array([False, False, False, False]),
                'techo': np.array([False, False]),
            },
        }

    def test_coverage_unions_stations_per_patch(self):
        self.assertEqual(
            coverage(self.stations[:2], self.flags, self.point_counts),
            {'capo': 100.0, 'techo': 100.0},
        )

    def test_combination_search_uses_iteration_order_after_equal_scores(self):
        best = best_combinations(self.stations, self.flags, self.point_counts, 2)
        self.assertEqual(best[1][1], (('a',),))
        self.assertEqual(best[2][1], (('a',), ('b',)))
        self.assertEqual(best[2][0][0], 100.0)

    def test_combination_search_breaks_minimum_coverage_ties_by_mean(self):
        stations = [('higher_mean',), ('balanced',)]
        flags = {
            ('higher_mean',): {
                'capo': np.array([True, True, True, True]),
                'techo': np.array([True, False]),
            },
            ('balanced',): {
                'capo': np.array([True, True, False, False]),
                'techo': np.array([True, False]),
            },
        }
        best = best_combinations(stations, flags, self.point_counts, 1)
        self.assertEqual(best[1][1], (('higher_mean',),))
        self.assertEqual(best[1][0], (50.0, 75.0))

    def test_recommendation_uses_smallest_count_meeting_target(self):
        best = best_combinations(self.stations, self.flags, self.point_counts, 2)
        chosen_k, chosen, meets_target = choose_recommendation(best, 90.0, 2)
        self.assertEqual(chosen_k, 2)
        self.assertEqual(chosen, (('a',), ('b',)))
        self.assertTrue(meets_target)

    def test_recommendation_marks_unmet_target(self):
        best = best_combinations(self.stations, self.flags, self.point_counts, 2)
        chosen_k, _, meets_target = choose_recommendation(best, 101.0, 2)
        self.assertEqual(chosen_k, 2)
        self.assertFalse(meets_target)


if __name__ == '__main__':
    unittest.main()
