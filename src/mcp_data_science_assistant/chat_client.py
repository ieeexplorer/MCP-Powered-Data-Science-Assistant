from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = PROJECT_ROOT / "src" / "mcp_data_science_assistant" / "server.py"


class MCPDataScienceChatClient:
    """Minimal terminal chat client that lets Claude call MCP tools."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    async def connect(self) -> None:
        server_params = StdioServerParameters(
            command="python",
            args=[str(SERVER_PATH)],
            env=os.environ.copy(),
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()

        tools_response = await self.session.list_tools()
        print("\nConnected tools:")
        for tool in tools_response.tools:
            print(f" - {tool.name}")

    async def process_query(self, query: str) -> str:
        if self.session is None:
            raise RuntimeError("Client is not connected.")

        tools_response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools_response.tools
        ]

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": query,
            }
        ]

        final_text: list[str] = []

        while True:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=1200,
                messages=messages,
                tools=available_tools,
            )

            assistant_message_content: list[Any] = []

            tool_used = False
            for content in response.content:
                assistant_message_content.append(content)

                if content.type == "text":
                    final_text.append(content.text)

                if content.type == "tool_use":
                    tool_used = True
                    result = await self.session.call_tool(content.name, content.input)

                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message_content,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": result.content,
                                }
                            ],
                        }
                    )
                    break

            if not tool_used:
                break

        return "\n".join(part for part in final_text if part).strip()

    async def close(self) -> None:
        await self.exit_stack.aclose()


async def _run() -> None:
    client = MCPDataScienceChatClient()
    await client.connect()

    print("\nType 'exit' to quit.")
    try:
        while True:
            user_query = input("\nYou: ").strip()
            if user_query.lower() in {"exit", "quit"}:
                break
            answer = await client.process_query(user_query)
            print(f"\nAssistant:\n{answer}")
    finally:
        await client.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
