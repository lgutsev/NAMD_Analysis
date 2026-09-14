import unittest
from unittest import mock

from namd_analysis import dispatch


class DispatchTests(unittest.TestCase):
    def test_character_command_routes_to_character_cli(self):
        with mock.patch("namd_analysis.dispatch.character_cli.main", return_value=7) as target:
            result = dispatch.main(["character-populations", "--sentinel"])
        self.assertEqual(result, 7)
        target.assert_called_once_with(["--sentinel"])

    def test_ordinary_command_routes_to_existing_cli(self):
        with mock.patch("namd_analysis.dispatch.cli.main", return_value=3) as target:
            result = dispatch.main(["inventory", "somewhere"])
        self.assertEqual(result, 3)
        target.assert_called_once_with(["inventory", "somewhere"])


if __name__ == "__main__":
    unittest.main()
