from contextlib import AsyncExitStack
import asyncio
import sys
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# 🚀 Minimal MCP Client Class
# Handles connecting to an MCP server, managing session,
# and exposing simple methods for tools, prompts, and resources.
class MCPClient:
    def __init__(self, url: str):
        # The MCP server URL (example: "http://127.0.0.1:8000/mcp/")
        self.url = url
        # AsyncExitStack allows us to manage multiple async context managers cleanly
        self.stack = AsyncExitStack()
        # Will hold our active MCP session
        self._session = None

    async def __aenter__(self):
        # Open a connection to the MCP server (read + write streams)
        read, write, _ = await self.stack.enter_async_context(
            streamablehttp_client(self.url)
        )
        # Create a new MCP client session over that connection
        self._session = await self.stack.enter_async_context(ClientSession(read, write))
        # Initialize the session (handshake + capabilities exchange)
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # Cleanly close all managed async contexts
        await self.stack.aclose()

    # 📌 List all available tools from the MCP server
    async def tool_list(self):
        return (await self._session.list_tools()).tools

    # 📌 Call a tool by name
    async def call_tool(self, name):
        return await self._session.call_tool(name)

    # 📌 List all available prompts
    async def list_prompt(self):
        return await self._session.list_prompts()

    # 📌 Get a specific prompt by name
    async def get_prompt(self, name):
        return await self._session.get_prompt(name)

    # 📌 Read resources by URI (documents, data, etc.)
    async def read_resources(self, uri):
        return await self._session.read_resource(uri)


# ✅ Example usage of the MCP client
async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <MCP_SERVER_URL>")
        sys.exit(1)

    mcp_server_url = sys.argv[1]

    # Connect to a local MCP server running on port 8000
    async with MCPClient(mcp_server_url) as client:
        tools = await client.tool_list()
        print(f"{len(tools)} tools are available!\n")
        for tool in tools:
            # Assuming each tool object has 'name' and optionally 'description' attributes
            description = getattr(tool, "description", "<no description>")
            print(f"######################")
            print(f"Tool - {description}")
            print() 


if __name__ == "__main__":
    asyncio.run(main())