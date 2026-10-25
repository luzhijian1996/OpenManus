import asyncio
import os
import sys
from typing import Dict

from app.tool.base import BaseTool


# 子进程包装器：从 stdin 读取代码执行，避免 Windows 命令行长度限制
_CODE_WRAPPER = "import sys; exec(compile(sys.stdin.read(), '<tool_code>', 'exec'))"


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"
    description: str = "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "执行超时时间（秒），默认 5。进行网络请求等耗时操作时请显式设置更大的值（如 60 或 120）。",
            },
        },
        "required": ["code"],
    }

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'observation' with execution output or error message and 'success' status.
        """
        # 使用全新子进程解释器执行代码，而非 multiprocessing spawn。
        # spawn 会重新导入父进程的 __main__ 模块（整个 app 链），
        # 在 Windows 上每次调用带来 5-7 秒的进程启动开销。
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _CODE_WRAPPER,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(input=code.encode("utf-8")), timeout=timeout
                )
            except asyncio.TimeoutError:
                # 超时：终止子进程并返回超时信息
                proc.kill()
                await proc.communicate()
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }

            output = stdout.decode("utf-8", errors="replace")
            return {
                "observation": output,
                "success": proc.returncode == 0,
            }
        except Exception as e:
            return {
                "observation": f"Failed to execute code: {str(e)}",
                "success": False,
            }
