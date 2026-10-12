"""Custom model loading and agent evaluation must be useful from a normal shell."""

import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from jevcache.cli import _agent_providers, _run_agents, main


class AgentFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_factory_keeps_custom_functions_unchanged(self):
        module = types.ModuleType("jac_test_models")

        async def generate(**kwargs):
            return kwargs

        decider = object()

        async def build():
            return {"generation": generate, "generation_identity": {"model": "custom"},
                    "decider": decider}

        module.build = build
        with patch.dict(sys.modules, {module.__name__: module}):
            supplied = await _agent_providers(argparse.Namespace(
                factory="jac_test_models:build"))
        self.assertIs(supplied["generation"], generate)
        self.assertIs(supplied["decider"], decider)

    async def test_factory_rejects_invalid_shape(self):
        module = types.ModuleType("jac_test_models")
        with patch.dict(sys.modules, {module.__name__: module}):
            for value in (None, {}, {"generation": lambda: None},
                          {"generation": lambda: None, "generation_identity": {},
                           "decider": object(), "unexpected": True}):
                module.build = lambda supplied=value: supplied
                with self.subTest(value=value), self.assertRaises(ValueError):
                    await _agent_providers(argparse.Namespace(factory="jac_test_models:build"))

    async def test_factory_requires_module_function_syntax(self):
        for name in ("models", ":build", "models:", "models:build:extra"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                await _agent_providers(argparse.Namespace(factory=name))

    async def test_callable_model_resources_are_drained_and_closed_on_evaluation_failure(self):
        lifecycle = []

        class Generator:
            async def __call__(self, **kwargs):
                return kwargs

            async def drain(self):
                lifecycle.append("drain")

            def close(self):
                lifecycle.append("close")

        supplied = {"generation": Generator(), "generation_identity": {"model": "test"},
                    "decider": object()}
        arguments = argparse.Namespace(offline=False, retrieval="bm25", mode="cache_first",
                                       repeats=1)
        with patch("jevcache.cli._agent_providers", AsyncMock(return_value=supplied)), \
             patch("jevcache.agent_evaluate.run_agent_evaluation", AsyncMock(
                 side_effect=ValueError("bad model configuration"))):
            with self.assertRaises(ValueError):
                await _run_agents(arguments)
        self.assertEqual(lifecycle, ["drain", "close"])


class AgentCommandTests(unittest.TestCase):
    def report(self, *, errors=0, unsafe=0):
        return {"fixture": True, "runs": {"no_cache": {
            "p50_ms": 1.0, "p95_ms": 2.0, "generation_calls_per_request": 1.0,
            "tool_calls": 2, "incorrect_responses": errors, "unsafe_reuses": unsafe,
            "setup_incorrect_responses": 0, "calibration_incorrect_decisions": 0,
        }}}

    def test_command_saves_diagnostics_and_returns_failure_for_errors(self):
        for errors in (0, 1):
            with self.subTest(errors=errors), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "agents.json"
                report = self.report(errors=errors)
                with patch("jevcache.cli._run_agents", AsyncMock(return_value=report)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    status = main(["evaluate-agents", "--offline", "--output", str(output)])
                self.assertEqual(status, errors)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_factory_exception_does_not_print_provider_secrets(self):
        printed = io.StringIO()
        with patch("jevcache.cli._run_agents", AsyncMock(
            side_effect=RuntimeError("Authorization: Bearer test-secret"))), \
             contextlib.redirect_stdout(printed):
            status = main(["evaluate-agents", "--factory", "models:build"])
        self.assertEqual(status, 1)
        self.assertNotIn("test-secret", printed.getvalue())
        self.assertIn("RuntimeError", printed.getvalue())

    def test_isolation_failure_returns_failure_even_if_answer_is_coincidentally_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.report()
            report["runs"]["no_cache"]["invalidation_failures"] = 1
            with patch("jevcache.cli._run_agents", AsyncMock(return_value=report)), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = main(["evaluate-agents", "--offline", "--output",
                               str(Path(directory) / "report.json")])
            self.assertEqual(status, 1)

    def test_conservative_rejection_is_reported_without_becoming_an_unsafe_reuse_failure(self):
        for false_positives, expected_status in ((0, 0), (1, 1)):
            with self.subTest(false_positives=false_positives), \
                 tempfile.TemporaryDirectory() as directory:
                report = self.report()
                row = report["runs"]["no_cache"]
                row.update(calibration_false_negatives=6,
                           calibration_false_positives=false_positives)
                output = io.StringIO()
                with patch("jevcache.cli._run_agents", AsyncMock(return_value=report)), \
                     contextlib.redirect_stdout(output):
                    status = main(["evaluate-agents", "--offline", "--output",
                                   str(Path(directory) / "report.json")])
                self.assertEqual(status, expected_status)
                self.assertIn(f"{false_positives}/6", output.getvalue())

    def test_source_selection_is_explicit(self):
        for arguments in ([], ["--offline", "--factory", "models:build"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(["evaluate-agents", *arguments])
                self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
