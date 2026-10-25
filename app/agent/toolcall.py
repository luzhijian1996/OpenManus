import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""

        # 优化：移除旧的 next_step_prompt，避免消息列表中重复元素过多
        if self.next_step_prompt and len(self.messages) > 0:
            # 从后往前查找并移除旧的 next_step_prompt
            for i in range(len(self.messages) - 1, -1, -1):
                if (self.messages[i].role == "user" and
                        self.messages[i].content == self.next_step_prompt):
                    self.messages.pop(i)
                    break

        # 添加最新的 next_step_prompt
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        # 打印 LLM 调用的详细信息
        logger.debug(f"\n{'='*50}")
        logger.debug(f"📝 LLM Call - System Prompt:")
        logger.debug(f"{'='*50}")
        if self.system_prompt:
            logger.debug(f"{self.system_prompt}")
        else:
            logger.debug("[No system prompt]")

        logger.debug(f"\n{'='*50}")
        logger.debug(f"💬 LLM Call - Messages ({len(self.messages)} total):")
        logger.debug(f"{'='*50}")
        for i, msg in enumerate(self.messages, 1):
            role = msg.role.upper()
            content_preview = msg.content[:200] + "..." if len(str(msg.content)) > 200 else msg.content
            logger.debug(f"[{i}] {role}: {content_preview}")
            # assistant 消息的 content 与 tool_calls 并存，一并打印便于调试
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_preview = str(tc.function.arguments)[:150]
                    logger.debug(f"    🔗 tool_call → {tc.function.name}({args_preview})")

        logger.debug(f"\n{'='*50}")
        logger.debug(f"🛠️ Available Tools: {[tool.name for tool in self.available_tools.tools]}")
        logger.debug(f"🎯 Tool Choice: {self.tool_choices}")
        logger.debug(f"{'='*50}\n")

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"达到最大 token 限制，无法继续执行: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # 直接将 LLM 输出的 content 与 tool_calls 合并加入消息
            # （content 可能为空，这是合法的 API 行为，不伪造填充）
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"处理过程中遇到错误: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """并行执行工具调用并处理结果"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        # 并行执行所有工具调用
        async def _execute_one(command: ToolCall):
            """执行单个工具调用，返回 (result_str, base64_image)"""
            result = await self.execute_tool(command)
            if self.max_observe:
                result = result[: self.max_observe]
            # 获取该工具产生的截图（如果有）
            image = getattr(self, '_tool_images', {}).get(command.id)
            return result, image

        # 初始化截图收集容器
        self._tool_images = {}

        # 使用 asyncio.gather 并行执行所有工具
        task_results = await asyncio.gather(
            *[_execute_one(cmd) for cmd in self.tool_calls],
            return_exceptions=True
        )

        results = []
        image_messages = []
        for i, (command, task_result) in enumerate(zip(self.tool_calls, task_results)):
            # 处理异常情况
            if isinstance(task_result, Exception):
                result = f"Error: Tool '{command.function.name}' failed: {task_result}"
                base64_img = None
                logger.error(f"🚨 Tool '{command.function.name}' raised exception: {task_result}")
            else:
                result, base64_img = task_result

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
            )
            self.memory.add_message(tool_msg)

            if base64_img:
                image_messages.append(
                    Message.user_message(
                        content=f"Image returned by {command.function.name}:",
                        base64_image=base64_img,
                    )
                )
            results.append(result)

        if image_messages:
            self.memory.add_messages(image_messages)

        # 清理临时截图容器
        self._tool_images = {}

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            logger.debug(f"🔧 Tool arguments: {args}")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                image_size_kb = len(result.base64_image) * 3 / 4 / 1024
                logger.info(f"📷 Tool '{name}' returned screenshot: {image_size_kb:.2f} KB")
                # 将截图存储到临时容器，供 act() 并行收集
                if not hasattr(self, '_tool_images'):
                    self._tool_images = {}
                self._tool_images[command.id] = result.base64_image
            else:
                logger.debug(f"📷 Tool '{name}' did not return screenshot")

            # 调试信息：显示工具执行结果
            if hasattr(result, "error") and result.error:
                logger.error(f"❌ Tool '{name}' failed: {result.error}")
            else:
                result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                logger.debug(f"✅ Tool '{name}' result preview: {result_preview}")

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
