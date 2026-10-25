import argparse
import asyncio
import sys

from app.agent.manus import Manus
from app.logger import logger

"""交互式运行Manus 通用代理 入口"""
async def main():
    # 确保提示符等 stdout 输出在管道环境下（如 PyCharm Run 窗口）立即显示
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Manus agent with a prompt")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    args = parser.parse_args()

    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        # Use command line prompt if provided, otherwise ask for input
        if args.prompt:
            prompt = args.prompt
        else:
            # 显式输出提示符并刷新，避免 stdout 缓冲导致提示符不显示
            print("Enter your prompt: ", end="", flush=True)
            prompt = input()
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
