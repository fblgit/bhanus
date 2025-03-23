# src/mcp_client/client.py

import json
from typing import List, Dict
from jsonrpcclient import request, parse, Ok, Error

class MCPClient:
    def __init__(self):
        """Initialize the MCP client with no connection."""
        self.server_url = None

    def connect(self, server_url: str) -> None:
        """
        Establish a connection to the MCP server.

        Args:
            server_url (str): URL of the MCP server (e.g., "http://localhost:5000").

        Raises:
            ValueError: If the server URL is invalid or connection fails.
        """
        self.server_url = server_url
        # Test connection by sending a simple request (e.g., list_tools)
        try:
            self.list_tools()
        except Exception as e:
            raise ValueError(f"Failed to connect to MCP server at {server_url}: {e}")

    def list_tools(self) -> List[Dict]:
        """
        Retrieve the list of available tools from the MCP server.

        Returns:
            List[Dict]: List of tools, each with 'name', 'description', 'input', and 'output'.

        Raises:
            RuntimeError: If the request fails or the response is invalid.
        """
        if not self.server_url:
            raise RuntimeError("Not connected to any MCP server. Call connect first.")

        response = request(self.server_url, "list_tools")
        parsed = parse(response)

        if isinstance(parsed, Ok):
            return parsed.result
        elif isinstance(parsed, Error):
            raise RuntimeError(f"Error listing tools: {parsed.message}")
        else:
            raise RuntimeError("Invalid response from server")

    def invoke_tool(self, tool_name: str, params: Dict) -> Dict:
        """
        Invoke a specific tool on the MCP server with the given parameters.

        Args:
            tool_name (str): Name of the tool to invoke.
            params (Dict): Parameters required by the tool.

        Returns:
            Dict: Result of the tool invocation.

        Raises:
            RuntimeError: If the request fails or the response is invalid.
        """
        if not self.server_url:
            raise RuntimeError("Not connected to any MCP server. Call connect first.")

        # Prepare the JSON-RPC request
        response = request(self.server_url, tool_name, params=params)
        parsed = parse(response)

        if isinstance(parsed, Ok):
            return parsed.result
        elif isinstance(parsed, Error):
            raise RuntimeError(f"Error invoking tool '{tool_name}': {parsed.message}")
        else:
            raise RuntimeError("Invalid response from server")

# Example usage for testing
if __name__ == "__main__":
    client = MCPClient()
    client.connect("http://localhost:5000")  # Assuming a local MCP server is running
    tools = client.list_tools()
    print("Available tools:", tools)
    if tools:
        result = client.invoke_tool(tools[0]["name"], {})
        print(f"Result from {tools[0]['name']}:", result)
