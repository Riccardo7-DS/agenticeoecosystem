"""Verify unhealthy dependencies prevent benchmark generation."""

import asyncio
import sys
import unittest
from types import ModuleType
from unittest.mock import AsyncMock, patch

from agenticeoecosystem.utils.healthcheck import check_mcp_toolset, require_healthy_services


class StartupHealthcheckTests(unittest.TestCase):
    """Exercise startup failures without contacting live benchmark services."""

    def test_failed_services_are_named(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MCP: mooc.*VLM endpoint"):
            require_healthy_services({"MCP: cdse": True, "MCP: mooc": False, "VLM endpoint": False})

    def test_healthy_services_are_accepted(self) -> None:
        require_healthy_services({"MCP: mooc": True, "VLM endpoint": True})

    def test_empty_mcp_tool_list_fails(self) -> None:
        toolset = ModuleType("toolset")
        toolset.get_tools = AsyncMock(return_value=[])
        self.assertFalse(asyncio.run(check_mcp_toolset("mooc", toolset)))

    def test_multi_generation_aborts_before_task_execution(self) -> None:
        import agenticeoecosystem.agents
        from agenticeoecosystem.evaluation.dataclasses import TaskSet
        from agenticeoecosystem.evaluation.runners.pipeline_runner import generate_solutions

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch.dict(sys.modules, {"agenticeoecosystem.agents.multi_agent": ModuleType("fake_agents")}), \
                 patch("agenticeoecosystem.utils.healthcheck.run_preflight_healthcheck",
                       AsyncMock(return_value={"MCP: mooc": False})), \
                 patch("agenticeoecosystem.evaluation.runners.pipeline_runner.run_pipeline") as execute:
                with self.assertRaisesRegex(RuntimeError, "MCP: mooc"):
                    generate_solutions(TaskSet(tasks=[]), model="test", temperature=0, pipeline="multi")
                execute.assert_not_called()
        finally:
            loop.close()
            asyncio.set_event_loop(None)


if __name__ == "__main__":
    unittest.main()
