import os
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.llm import LLM, MULTIMODAL_MODELS
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.web_search import WebSearch


_BROWSER_USE_SERVER_ID = "browser_use"
_BROWSER_USE_COMMAND = "uvx"
_BROWSER_USE_ARGS = ["browser-use", "--cli-mcp"]
_BROWSER_USE_ENV_VARS = (
    "BROWSER_USE_API_KEY",
    "BROWSER_USE_CLOUD_API_URL",
    "BU_BROWSER_ID",
    "BU_CDP_URL",
    "BU_CDP_WS",
    "BU_NAME",
)
_BROWSER_USE_TRANSPORT_INSTRUCTIONS = """\
Browser Use CLI 3.0 is exposed here as MCP tools. When the Browser Use skill
shows `browser-use <<'PY'`, pass the Python body to `browser_exec` instead.
Use `browser_screenshot` when visual inspection is needed. Both tools use the
same persistent browser-harness session as CLI 3.0.
"""


def _browser_use_env() -> Dict[str, str]:
    return {name: value for name in _BROWSER_USE_ENV_VARS if (value := os.getenv(name))}


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            WebSearch(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    mcp_instruction_servers: set[str] = Field(default_factory=set, exclude=True)
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """同步初始化基本组件。"""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        # 跳过自动连接的条件：
        # 1. 已在 mcp.json 中配置了 browser_use 服务器（走下方的手动配置循环）
        # 2. 环境变量 OPENMANUS_DISABLE_BROWSER_USE 被设置
        # 3. config.toml [mcp] 段配置了 disable_browser_use = true
        disable_by_env = os.getenv(
            "OPENMANUS_DISABLE_BROWSER_USE", ""
        ).lower() in {"1", "true", "yes"}
        if (
            _BROWSER_USE_SERVER_ID not in config.mcp_config.servers
            and not config.mcp_config.disable_browser_use
            and not disable_by_env
        ):
            try:
                await self.connect_mcp_server(
                    _BROWSER_USE_COMMAND,
                    _BROWSER_USE_SERVER_ID,
                    use_stdio=True,
                    stdio_args=_BROWSER_USE_ARGS,
                    tool_name_prefix=False,
                    stdio_env=_browser_use_env(),
                )
                logger.info("Connected to Browser Use CLI 3.0 through MCP")
            except Exception as e:
                logger.error(f"Failed to connect to Browser Use CLI 3.0: {e}")
        # 禁用时静默跳过，不打印日志（保持与老项目一致的启动输出）

        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                            tool_name_prefix=server_id != _BROWSER_USE_SERVER_ID,
                            stdio_env=(
                                _browser_use_env()
                                if server_id == _BROWSER_USE_SERVER_ID
                                else None
                            ),
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: Optional[List[str]] = None,
        tool_name_prefix: bool = True,
        stdio_env: Optional[Dict[str, str]] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url,
                stdio_args or [],
                server_id,
                tool_name_prefix=tool_name_prefix,
                env=stdio_env,
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # Update available tools with only the new tools from this server
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

        resolved_server_id = server_id or server_url
        instructions = self.mcp_clients.server_instructions.get(resolved_server_id)
        if instructions and resolved_server_id not in self.mcp_instruction_servers:
            transport_instructions = (
                f"{_BROWSER_USE_TRANSPORT_INSTRUCTIONS}\n"
                if resolved_server_id == _BROWSER_USE_SERVER_ID
                else ""
            )
            self.memory.add_message(
                Message.system_message(
                    f"{transport_instructions}MCP server instructions:\n{instructions}"
                )
            )
            self.mcp_instruction_servers.add(resolved_server_id)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """清理 Manus agent 资源。"""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # 仅在已初始化的情况下断开所有 MCP 服务器连接
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    async def think(self) -> bool:
        """处理当前状态，并在适当的上下文中决定下一步行动。"""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        original_prompt = self.next_step_prompt
        original_llm = self.llm

        # 检查最近消息中是否有浏览器工具在使用
        recent_messages = self.memory.messages[-5:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if hasattr(msg, 'tool_calls') and msg.tool_calls
            for tc in msg.tool_calls
        )

        # 检查消息中是否有浏览器截图（说明浏览器工具已使用，需要视觉理解）
        has_browser_screenshot = any(
            (isinstance(msg, Message) and getattr(msg, 'base64_image', None))
            or (isinstance(msg, dict) and msg.get("base64_image"))
            for msg in recent_messages
        )

        # 智能切换策略：
        # browser-use 库返回的元素信息格式：[index]<type>text</type>
        # 包含：索引、元素类型（button、input等）、文本描述
        # 如果元素描述足够详细，默认模型应该能够根据文本匹配选择正确的元素
        # 因此：默认使用快速模型，只在有截图且当前模型不支持图片时才切换到视觉模型

        # 只有浏览器工具在最近步骤中实际被使用时，才采集浏览器状态并格式化 prompt
        # 避免每步无条件调用 get_current_state()（截图 + DOM 解析约 1 秒）
        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        # === 视觉模型自动切换逻辑 ===
        # 当消息中包含浏览器截图，且当前模型不支持图片理解时，自动切换到 vision 模型
        if has_browser_screenshot and self.llm.model not in MULTIMODAL_MODELS:
            if "vision" in config.llm:
                vision_llm = LLM(config_name="vision")
                if vision_llm.model in MULTIMODAL_MODELS:
                    logger.info(
                        f"👁️ Screenshots detected & current model '{self.llm.model}' "
                        f"does NOT support images. Switching to vision model: '{vision_llm.model}'"
                    )
                    self.llm = vision_llm
                else:
                    logger.warning(
                        f"⚠️ Vision model '{vision_llm.model}' is NOT in MULTIMODAL_MODELS. "
                        f"Images will be stripped. Please add it to MULTIMODAL_MODELS in llm.py."
                    )
            else:
                logger.warning(
                    f"⚠️ Screenshots detected but no [llm.vision] configured in config.toml. "
                    f"Images will be ignored by model '{self.llm.model}'."
                )
        elif has_browser_screenshot and self.llm.model in MULTIMODAL_MODELS:
            pass  # 当前模型已支持图片，无需切换

        result = await super().think()

        # 恢复原始 prompt
        self.next_step_prompt = original_prompt

        # 恢复原始 LLM 的逻辑：
        # 只有当最近的消息中既没有浏览器工具调用，也没有截图时，才恢复
        if original_llm != self.llm:
            current_browser_in_use = any(
                tc.function.name == BrowserUseTool().name
                for msg in self.memory.messages[-3:]
                if hasattr(msg, 'tool_calls') and msg.tool_calls
                for tc in msg.tool_calls
            )
            current_has_screenshot = any(
                (isinstance(msg, Message) and getattr(msg, 'base64_image', None))
                or (isinstance(msg, dict) and msg.get("base64_image"))
                for msg in self.memory.messages[-3:]
            )
            if not current_browser_in_use and not current_has_screenshot:
                logger.debug(f"🔄 Restoring original LLM: {original_llm.model}")
                self.llm = original_llm

        return result
