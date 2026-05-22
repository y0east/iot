import unittest

from iot_servo_tracker.common.query import canonical_detection_query, query_matches_class


class QueryTests(unittest.TestCase):
    def test_person_aliases_are_canonicalized(self) -> None:
        self.assertEqual(canonical_detection_query("people"), "person")
        self.assertEqual(canonical_detection_query("see man"), "person")
        self.assertEqual(canonical_detection_query("track the woman"), "person")

    def test_specific_queries_are_preserved(self) -> None:
        self.assertEqual(canonical_detection_query("red cup"), "red cup")

    def test_query_matches_detector_class_without_substring_false_positive(self) -> None:
        self.assertTrue(query_matches_class("people", "person"))
        self.assertTrue(query_matches_class("see man", "person"))
        self.assertTrue(query_matches_class("red cup", "cup"))
        self.assertFalse(query_matches_class("red carpet", "car"))


if __name__ == "__main__":
    unittest.main()
