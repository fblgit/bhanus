# ./src/mcp_client/client.py
import json
import logging
import os
import requests
import yaml
from typing import List, Dict, Any
from jsonrpcclient import request, parse, Ok, Error, request_json
from tenacity import retry, stop_after_attempt, wait_exponential

#from src.inference.vllm_engine import VLLMEngine
#from src.generation.interceptor import GenerationInterceptor
from src.prompts.utils import load_prompt_template, format_prompt
from src.generation.parser import parse_agent_output

# Configure logging
logger = logging.getLogger(__name__)

class MCPClient:
    """Client for interacting with an MCP server via JSON-RPC 2.0 with tool registration and agent execution."""

    def __init__(self, config_path: str = "config/config.yaml", engine: Any = None,
                 interceptor: Any = None, verbose: bool = False, log_level: str = "INFO"):
        """
        Initialize the MCP client with a persistent HTTP session, engine, and interceptor.

        Args:
            config_path (str): Path to the configuration file (default: "config/config.yaml").
            engine (VLLMEngine, optional): Engine for agent tool generation and registry access.
            interceptor (GenerationInterceptor, optional): Interceptor for processing agent streams.
            verbose (bool): If True, log detailed client operations.
            log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING").
        """
        self.server_url: str = ""
        self.session = requests.Session()
        self.engine = engine
        self.interceptor = interceptor
        self.verbose = verbose
        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or "mcp" not in config:
            raise ValueError("Invalid config.yaml: must contain 'mcp' section")
        self.function_registry_file = config.get("registry", {}).get("path", "function_registry.json")
        self.default_url = f"http://{config['mcp'].get('host', '0.0.0.0')}:{config['mcp'].get('port', 6000)}"
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                            format="%(asctime)s - %(levelname)s - %(message)s")
        if self.verbose:
            logger.debug("Initialized MCPClient with registry path: %s, default URL: %s, %s engine, %s interceptor",
                         self.function_registry_file, self.default_url, "active" if engine else "no",
                         "active" if interceptor else "no")

    def connect(self, server_url: str = None) -> None:
        """
        Set and validate the MCP server URL for subsequent requests, defaulting to local server.

        Args:
            server_url (str, optional): URL of the MCP server (e.g., "http://localhost:6000"). Defaults to config value.
        """
        server_url = server_url or self.default_url
        if not server_url or not isinstance(server_url, str):
            logger.error("Invalid server_url provided: %s", server_url)
            raise ValueError("server_url must be a non-empty string")

        try:
            if not server_url.startswith(("http://", "https://")):
                raise ValueError("server_url must start with http:// or https://")
            self.server_url = server_url.rstrip('/')
            response = self.session.get(f"{self.server_url}/", timeout=5)
            if response.status_code >= 500:
                raise ValueError(f"Server at {self.server_url} returned HTTP {response.status_code}")
            if self.verbose:
                logger.info("Connected to MCP server at %s", self.server_url)
        except requests.RequestException as e:
            error_msg = f"Failed to connect to server '{server_url}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def list_tools(self, include_local: bool = False) -> List[Dict]:
        """
        Retrieve and validate the list of available tools from the MCP server, optionally with local registry.

        Args:
            include_local (bool): If True, merge with locally registered functions from engine or file.

        Returns:
            List[Dict]: List of tools with 'name', 'description', 'input', 'output', and 'type' keys.
        """
        if not self.server_url:
            logger.error("Not connected to any MCP server")
            raise RuntimeError("Not connected to any MCP server. Call connect first.")

        try:
            if self.verbose:
                logger.info("Requesting list of tools from MCP server at %s", self.server_url)
            rpc_request = request("list_tools") #, session=self.session)
            response = self.session.post(self.server_url, json=rpc_request, timeout=5, headers={'Content-Type': 'application/json'})
            parsed = parse(response.json())

            if isinstance(parsed, Ok):
                tools = parsed.result
                if not isinstance(tools, list):
                    raise RuntimeError(f"Expected list of tools, got: {type(tools)}")
                required_fields = {"name", "description", "input", "output"}
                validated_tools = []
                for tool in tools:
                    if not isinstance(tool, dict) or not required_fields.issubset(tool.keys()):
                        logger.warning("Tool missing required fields: %s", tool)
                        continue
                    tool["type"] = tool.get("type", "code")
                    validated_tools.append(tool)
                if not validated_tools:
                    raise RuntimeError("No valid tools returned matching MCP schema")

                if include_local and self.engine:
                    local_tools = self.engine.list_registered_functions()
                    tool_names = {tool["name"] for tool in validated_tools}
                    for local_tool in local_tools:
                        if local_tool["name"] not in tool_names:
                            validated_tools.append(local_tool)
                    if self.verbose:
                        logger.debug("Merged local tools from engine: %s", local_tools)
                elif include_local:
                    local_tools = self._get_local_tools()
                    tool_names = {tool["name"] for tool in validated_tools}
                    for local_tool in local_tools:
                        if local_tool["name"] not in tool_names:
                            validated_tools.append(local_tool)
                    if self.verbose:
                        logger.debug("Merged local tools from file: %s", local_tools)

                if self.verbose:
                    logger.debug("Received and validated tools: %s", validated_tools)
                return validated_tools
            elif isinstance(parsed, Error):
                error_msg = f"Error listing tools: {parsed.message}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.error("Invalid response from server: %s", parsed)
                raise RuntimeError("Invalid response from server")

        except Exception as e:
            error_msg = f"Failed to list tools after retries: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def invoke_tool(self, tool_name: str, params: Dict, consent: bool = False) -> Dict:
        """
        Invoke a specific tool on the MCP server or locally if agent-based, requiring user consent.

        Args:
            tool_name (str): Name of the tool to invoke (e.g., "search_from_api").
            params (Dict): Parameters required by the tool.
            consent (bool): User consent flag, required per MCP spec.

        Returns:
            Dict: Result of the tool invocation.
        """
        if not self.server_url:
            logger.error("Not connected to any MCP server")
            raise RuntimeError("Not connected to any MCP server. Call connect first.")
        if not tool_name or not isinstance(tool_name, str):
            logger.error("Invalid tool_name provided: %s", tool_name)
            raise RuntimeError("tool_name must be a non-empty string")
        if not isinstance(params, dict):
            logger.error("Invalid params provided: %s", params)
            raise RuntimeError("params must be a dictionary")
        if not consent:
            logger.error("User consent required for tool '%s'", tool_name)
            raise RuntimeError("User consent required for tool invocation per MCP specification")

        try:
            if self.verbose:
                logger.info("Invoking tool '%s' with params: %s and consent: %s", tool_name, params, consent)
            # Check if tool is agent-based and executable locally
            local_tools = self.engine.list_registered_functions() if self.engine else self._get_local_tools()
            tool_def = next((t for t in local_tools if t["name"] == tool_name), None)
            if tool_def and tool_def.get("type") == "agent" and self.engine and self.interceptor:
                return self._invoke_agent_tool(tool_name, params, tool_def)

            # Server invocation
            rpc_request = request(
                tool_name,
                params={"consent": consent, **params},
                #session=self.session
            )
            response = self.session.post(self.server_url, json=rpc_request, timeout=5, headers={'Content-Type': 'application/json'})
            parsed = parse(response.json())
            #parsed = parse(response)

            if isinstance(parsed, Ok):
                result = parsed.result
                if not isinstance(result, dict):
                    logger.warning("Tool '%s' returned non-dict result: %s", tool_name, result)
                    result = {"result": result}
                if self.verbose:
                    logger.debug("Tool '%s' result: %s", tool_name, result)
                return result
            elif isinstance(parsed, Error):
                error_msg = f"Error invoking tool '{tool_name}': {parsed.message}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.error("Invalid response from server: %s", parsed)
                raise RuntimeError("Invalid response from server")

        except Exception as e:
            error_msg = f"Failed to invoke tool '{tool_name}' with params {params} after retries: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def register_tool(self, name: str, definition: Dict) -> Dict:
        """
        Register a new tool with the MCP server and local storage, returning status.
    
        Args:
            name (str): Name of the tool (e.g., "tools_picker").
            definition (Dict): Tool details (e.g., {"description": "...", "input": {}, "output": "...", "type": "..."}).
    
        Returns:
            Dict: Status of registration (e.g., {"status": "success", "message": "..."}).
        """
        if not self.server_url:
            logger.error("Not connected to any MCP server")
            raise RuntimeError("Not connected to any MCP server. Call connect first.")
        if not name or not isinstance(name, str):
            logger.error("Invalid tool name provided: %s", name)
            raise RuntimeError("Tool name must be a non-empty string")
        if not isinstance(definition, dict):
            logger.error("Invalid definition provided: %s", definition)
            raise RuntimeError("Definition must be a dictionary")
    
        payload = {
            "name": name,
            "description": definition.get("description", f"Tool {name} created by model"),
            "input": definition.get("input", {}),
            "output": definition.get("output", "string"),
            "type": definition.get("type", "code")
        }
        if "output_format" in definition:
            payload["output_format"] = definition["output_format"]
    
        try:
            if self.verbose:
                logger.info("Registering tool '%s' with definition: %s", name, payload)
            # Adjusted call to jsonrpcclient.request
            rpc_request = request("register_tool", params=payload)
            response = self.session.post(self.server_url, json=rpc_request, timeout=5, headers={'Content-Type': 'application/json'})
            parsed = parse(response.json())
    
            if isinstance(parsed, Ok):
                # Local sync if engine is available
                if self.engine:
                    self.engine.register_function(name, payload)
                    if self.verbose:
                        logger.debug("Synced tool '%s' with local engine registry", name)
                else:
                    existing_tools = self._get_local_tools()
                    existing_tools = [t for t in existing_tools if t["name"] != name]
                    existing_tools.append(payload)
                    with open(self.function_registry_file, "w", encoding="utf-8") as f:
                        for tool in existing_tools:
                            json.dump(tool, f)
                            f.write("\n")
                    if self.verbose:
                        logger.debug("Synced tool '%s' with local file registry", name)
                if self.verbose:
                    logger.info("Successfully registered tool '%s' on server and locally", name)
                return {"status": "success", "message": f"Tool '{name}' registered successfully"}
            elif isinstance(parsed, Error):
                error_msg = f"Server error registering tool '{name}': {parsed.message}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            else:
                error_msg = f"Invalid server response for tool '{name}': {parsed}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
    
        except Exception as e:
            error_msg = f"Failed to register tool '{name}' after retries: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

    def _get_local_tools(self) -> List[Dict]:
        """Retrieve locally registered tools from the function registry file."""
        tools = []
        try:
            if os.path.exists(self.function_registry_file):
                with open(self.function_registry_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            tools.append(json.loads(line))
            return tools
        except Exception as e:
            logger.warning("Failed to read local tools: %s; returning empty list", e)
            return []

    def _invoke_agent_tool(self, tool_name: str, params: Dict, tool_def: Dict) -> Dict:
        """
        Execute an agent-based tool locally using VLLMEngine and GenerationInterceptor.

        Args:
            tool_name (str): Name of the agent tool (e.g., "tools_picker").
            params (Dict): Parameters for the agent (e.g., {"task": "summarize code"}).
            tool_def (Dict): Tool definition from the registry.

        Returns:
            Dict: Result of the agent execution.
        """
        if self.verbose:
            logger.info("Executing agent tool '%s' with params: %s", tool_name, params)

        try:
            # Load Jinja2 template for the agent
            template_name = tool_name
            template_info = load_prompt_template(template_name, verbose=self.verbose)
            tools = self.list_tools(include_local=True)
            prompt = format_prompt(template_info, task=params.get("task", ""), tools=tools, verbose=self.verbose)
        except Exception as e:
            logger.warning("Failed to load template for '%s': %s; using fallback prompt", tool_name, e)
            prompt = f"{tool_name.capitalize()} task: '{params.get('task', '')}' with tools: {json.dumps(tools)}"

        try:
            generator = self.engine.generate_with_interception(prompt, self.interceptor, max_tokens=200)
            result = "".join(list(generator)).strip()
            parsed_result = parse_agent_output(result, tool_name, engine=self.engine, verbose=self.verbose)

            if not parsed_result:
                error_msg = f"Agent '{tool_name}' returned invalid output: '{result}'"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if self.verbose:
                logger.debug("Agent '%s' result: %s", tool_name, parsed_result)
            return parsed_result

        except Exception as e:
            error_msg = f"Failed to execute agent tool '{tool_name}' with params {params}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

# Example usage for testing
if __name__ == "__main__":
    class MockVLLMEngine:
        def list_registered_functions(self):
            return [{"name": "tools_picker", "type": "agent", "output_format": "Selected tools: \\[.*\\]"}]
        def generate_with_interception(self, prompt: str, interceptor, **kwargs) -> List[str]:
            return ["Selected tools: [parse_html, summarise_text]"]
        def register_function(self, name: str, definition: Dict) -> None:
            logger.info("Mock: Registered %s with %s", name, definition)

    class MockInterceptor:
        def intercept_stream(self, generator):
            yield from generator

    engine = MockVLLMEngine()
    interceptor = MockInterceptor()
    client = MCPClient(config_path="config/config.yaml", engine=engine, interceptor=interceptor,
                       verbose=True, log_level="DEBUG")
    client.connect()

    # Test listing tools
    print("\nListing tools:")
    tools = client.list_tools(include_local=True)
    print("Available tools:", tools)

    # Test registering a tool
    print("\nRegistering a tool:")
    result = client.register_tool("tools_picker", {
        "description": "Selects tools for a given task",
        "input": {"task": "string"},
        "output": "list",
        "type": "agent"
    })
    print("Registration result:", result)

    # Test invoking an agent tool
    print("\nInvoking agent tool:")
    result = client.invoke_tool("tools_picker", {"task": "summarize code"}, consent=True)
    print("Agent result:", result)
