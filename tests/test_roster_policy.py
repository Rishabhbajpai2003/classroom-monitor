import unittest

from app.models.roster_policy import capped_reportable_ids


class RosterPolicyTests(unittest.TestCase):
    def test_named_students_take_priority_and_unknowns_fill_headroom(self):
        selected = capped_reportable_ids(
            ["STU_001", "STU_002"],
            [("TEMP_0001", 9.0), ("TEMP_0002", 12.0), ("TEMP_0003", 3.0)],
            roster_limit=4,
        )
        self.assertEqual(selected, {"STU_001", "STU_002", "TEMP_0002", "TEMP_0001"})

    def test_roster_cap_truncates_when_named_ids_already_fill_limit(self):
        selected = capped_reportable_ids(
            ["STU_001", "STU_002", "STU_003"],
            [("TEMP_0001", 99.0)],
            roster_limit=2,
        )
        self.assertEqual(selected, {"STU_001", "STU_002"})


if __name__ == "__main__":
    unittest.main()
