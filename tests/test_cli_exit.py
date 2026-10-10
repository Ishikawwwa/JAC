"""Module commands must expose evaluation failures to shells and CI."""

import runpy
import unittest
from unittest.mock import patch


class ModuleExitTests(unittest.TestCase):
    def test_evaluation_failure_propagates_from_both_module_commands(self):
        for module in ("jac", "jevcache"):
            with self.subTest(module=module), patch("jevcache.cli.main", return_value=1):
                with self.assertRaises(SystemExit) as caught:
                    runpy.run_module(module, run_name="__main__")
                self.assertEqual(caught.exception.code, 1)

    def test_success_remains_a_zero_exit(self):
        for module in ("jac", "jevcache"):
            with self.subTest(module=module), patch("jevcache.cli.main", return_value=0):
                with self.assertRaises(SystemExit) as caught:
                    runpy.run_module(module, run_name="__main__")
                self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
