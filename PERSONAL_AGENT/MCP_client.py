import asyncio
import json
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from logger_setup import get_logger

# Official MCP SDK imports
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = get_logger(__name__)

class MCPClient:
    """
    A unified MCP Client class to connect to multiple servers via SSE or STDIO.
    """
    def __init__(self):
        # Map of server_name -> ClientSession
        self.sessions: Dict[str, ClientSession] = {}
        self._exit_stack = AsyncExitStack()

    async def connect_sse(self, server_name: str, url: str):
        """Connect to an MCP server via Server-Sent Events (SSE)."""
        sse_transport = await self._exit_stack.enter_async_context(sse_client(url))
        read_stream, write_stream = sse_transport
        
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self.sessions[server_name] = session
        return session

    async def connect_stdio(self, server_name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """Connect to a local MCP server via standard input/output (STDIO)."""
        # Ensure env is dict with string values if provided
        safe_env = {k: str(v) for k, v in env.items()} if env else None
        
        server_params = StdioServerParameters(command=command, args=args, env=safe_env)
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self.sessions[server_name] = session
        return session

    async def load_from_config(self, config_path: str):
        """Load and connect to multiple STDIO servers from a config JSON."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        mcp_servers = config.get("mcpServers", {})
        for name, details in mcp_servers.items():
            if details.get("command"):
                logger.info(f"Connecting to STDIO server: {name}...")
                await self.connect_stdio(
                    server_name=name,
                    command=details["command"],
                    args=details.get("args", []),
                    env=details.get("env", None)
                )

    def get_session(self, server_name: str) -> ClientSession:
        if server_name not in self.sessions:
            raise RuntimeError(f"Server '{server_name}' is not connected.")
        return self.sessions[server_name]

    async def list_tools(self, server_name: str):
        """List all tools exposed by a specific MCP server."""
        session = self.get_session(server_name)
        response = await session.list_tools()
        return response.tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """Call a specific tool on a specific server with arguments."""
        session = self.get_session(server_name)
        result = await session.call_tool(tool_name, arguments=arguments)
        return result

    async def close(self):
        """Gracefully close all connections to all MCP servers."""
        await self._exit_stack.aclose()
        self.sessions.clear()

# Example usage script
async def main():
    client = MCPClient()
    try:
        # Load from the config file you created
        await client.load_from_config("server_config.json")
        logger.info("Successfully connected to servers in config!")
        
        # Iterate over all connected servers and list their tools
        for server_name in client.sessions.keys():
            tools = await client.list_tools(server_name)
            logger.info(f"[{server_name}] Available tools:")
            for t in tools:
                logger.info(f" - {t.name}: {t.description}")
                
    except Exception as e:
        logger.error(f"Failed to run MCP Client: {e}")
    finally:
        await client.close()
        logger.info("All connections closed.")

if __name__ == "__main__":
    asyncio.run(main())
