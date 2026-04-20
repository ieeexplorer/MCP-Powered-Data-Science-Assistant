"""Terminal client for chatting with the MCP data science server."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from anthropic import APIConnectionError, APIError, Anthropic, RateLimitError
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp_data_science_assistant.runtime import build_server_command
except ModuleNotFoundError:
    from runtime import build_server_command

load_dotenv()

MAX_RETRIES = 2


class MCPDataScienceChatClient:
    """Terminal chat client that lets Claude call MCP tools with retry handling."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env file.")
        self.anthropic = Anthropic(api_key=api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    async def connect(self) -> None:
        server_command, server_args = build_server_command()
        server_params = StdioServerParameters(
            command=server_command,
            args=server_args,
            env=os.environ.copy(),
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()

        tools_response = await self.session.list_tools()
        print("\nConnected to MCP server. Available tools:")
        for tool in tools_response.tools:
            description = tool.description or "No description provided."
            print(f" - {tool.name}: {description[:60]}...")

    async def _create_response(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=1200,
                    messages=messages,
                    tools=tools,
                )
            except RateLimitError:
                if attempt >= MAX_RETRIES:
                    raise
                wait_seconds = 2 ** attempt
                print(f"Rate limited. Retrying in {wait_seconds}s...")
                await asyncio.sleep(wait_seconds)
            except APIConnectionError:
                if attempt >= MAX_RETRIES:
                    raise
                wait_seconds = 2 ** attempt
                print(f"Connection issue. Retrying in {wait_seconds}s...")
                await asyncio.sleep(wait_seconds)

        raise RuntimeError("Failed to create Anthropic response after retries.")

    @staticmethod
    def _extract_structured_error(result: Any) -> str | None:
        if not getattr(result, "content", None):
            return None
        for item in result.content:
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "error" in payload:
                error_type = payload.get("error_type", "Error")
                return f"{error_type}: {payload['error']}"
        return None

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
            try:
                response = await self._create_response(messages, available_tools)
            except APIError as error:
                return f"Anthropic API error: {error}"

            assistant_message_content: list[Any] = []

            tool_used = False
            for content in response.content:
                assistant_message_content.append(content)

                if content.type == "text":
                    final_text.append(content.text)

                if content.type == "tool_use":
                    tool_used = True
                    print(f"Calling tool: {content.name}...")
                    result = await self.session.call_tool(content.name, content.input)
                    structured_error = self._extract_structured_error(result)
                    if structured_error:
                        print(f"Tool error: {structured_error}")

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
    print("\nMCP Data Science Assistant - Terminal Client")
    print("-------------------------------------------")
    try:
        client = MCPDataScienceChatClient()
        await client.connect()
    except Exception as error:
        print(f"\nFailed to start: {error}")
        return

    print("\nType your query, or 'exit' to quit.")
    print("Example: Analyze data/churn_sample.csv and tell me which feature matters most for churn.\n")
    try:
        while True:
            user_query = input("\nYou: ").strip()
            if user_query.lower() in {"exit", "quit"}:
                break
            if not user_query:
                continue
            print("Assistant is thinking...", end="", flush=True)
            answer = await client.process_query(user_query)
            print(f"\rAssistant:\n{answer}\n")
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await client.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
