import json
from typing import TYPE_CHECKING, Optional

from app.agent.mcp import MCPAgent
from app.logger import logger
from app.prompt.browser import NEXT_STEP_PROMPT
from app.tool.terminate import Terminate


# Avoid circular import if BrowserAgent needs BrowserContextHelper
if TYPE_CHECKING:
    from app.agent.base import BaseAgent  # Or wherever memory is defined


_SANDBOX_BROWSER_TOOL_NAME = "sandbox_browser"
_BROWSER_USE_TOOL_NAME = "browser_use"


class BrowserContextHelper:
    def __init__(self, agent: "BaseAgent"):
        self.agent = agent

    def _find_browser_tool(self):
        """两级回退查找浏览器工具：优先 BrowserUseTool，其次 SandboxBrowserTool。"""
        browser_tool = self.agent.available_tools.get_tool(_BROWSER_USE_TOOL_NAME)
        if not browser_tool:
            browser_tool = self.agent.available_tools.get_tool(_SANDBOX_BROWSER_TOOL_NAME)
        return browser_tool

    async def get_browser_state(self) -> Optional[dict]:
        """获取浏览器状态文本信息（URL、标签页、交互元素等）。

        注意：此方法不再采集截图。截图由 toolcall.py 的 act() 在工具执行后自动添加，
        避免同一页面状态被截图两次导致 token 浪费和 API 报错。
        """
        browser_tool = self._find_browser_tool()
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            return None
        try:
            result = await browser_tool.get_current_state()
            if result.error:
                logger.debug(f"Browser state error (skipping): {result.error}")
                return None
            return json.loads(result.output)
        except Exception as e:
            logger.debug(f"Failed to get browser state: {str(e)}")
            return None

    async def format_next_step_prompt(self) -> str:
        """获取浏览器状态并格式化浏览器提示词。

        仅采集文本状态信息（URL、标签页、交互元素），不发送截图。
        截图由 toolcall.py 的 act() 在工具执行后自动附加到消息中。
        """
        browser_state = await self.get_browser_state()
        url_info, tabs_info, content_above_info, content_below_info = "", "", "", ""
        results_info = ""

        if browser_state and not browser_state.get("error"):
            url_info = f"\n   URL: {browser_state.get('url', 'N/A')}\n   Title: {browser_state.get('title', 'N/A')}"
            tabs = browser_state.get("tabs", [])
            if tabs:
                tabs_info = f"\n   {len(tabs)} tab(s) available"
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)
            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"
            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            # 调试信息：显示可交互元素数量
            interactive_elements = browser_state.get("interactive_elements", "")
            element_count = interactive_elements.count("[") if interactive_elements else 0
            logger.debug(f"🔍 Browser state: {element_count} interactive elements, URL={browser_state.get('url', 'N/A')}")

        prompt = NEXT_STEP_PROMPT.format(
            url_placeholder=url_info,
            tabs_placeholder=tabs_info,
            content_above_placeholder=content_above_info,
            content_below_placeholder=content_below_info,
            results_placeholder=results_info,
        )

        # 追加交互元素列表到 prompt，让 LLM 直接看到可点击元素
        if browser_state and not browser_state.get("error"):
            interactive_elements = browser_state.get("interactive_elements", "")
            if interactive_elements:
                prompt += "\n\n[Current state starts here]\n"
                prompt += "Interactive Elements:\n"
                prompt += interactive_elements
                prompt += "\n"

        return prompt

    async def cleanup_browser(self):
        """清理浏览器资源。"""
        browser_tool = self._find_browser_tool()
        if browser_tool and hasattr(browser_tool, "cleanup"):
            await browser_tool.cleanup()


class BrowserAgent(MCPAgent):
    """Browser-only agent backed by the Browser Use CLI 3.0 MCP server."""

    name: str = "browser"
    description: str = "A browser agent that can control a browser to accomplish tasks"
    system_prompt: str = """\
Use Browser Use CLI 3.0 through `browser_exec`. Pass the Python body from CLI
examples as the `code` argument. Use `browser_screenshot` when visual context
is needed. The browser-harness session persists across calls. Use `terminate`
when the task is complete.
"""

    async def initialize(self) -> None:
        await super().initialize(
            connection_type="stdio",
            command="uvx",
            args=["browser-use", "--cli-mcp"],
            server_id="browser_use",
            tool_name_prefix=False,
        )
        self.available_tools.add_tool(Terminate())

    @classmethod
    async def create(cls, **kwargs) -> "BrowserAgent":
        instance = cls(**kwargs)
        await instance.initialize()
        return instance

    async def run(self, request: Optional[str] = None) -> str:
        if not self.mcp_clients.sessions:
            await self.initialize()
        return await super().run(request)
