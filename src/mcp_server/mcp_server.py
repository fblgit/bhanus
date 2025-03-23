# ./src/mcp_server/mcp_server.py
import json
import logging
import os
from typing import Dict, List, Any
from flask import Flask, request, jsonify
import portalocker
import yaml

#from src.inference.vllm_engine import VLLMEngine
from src.generation.interceptor import GenerationInterceptor
from src.prompts.utils import load_prompt_template, format_prompt
from src.generation.parser import parse_agent_output
from src.handlers import HANDLER_REGISTRY

# Configure logging
logger = logging.getLogger(__name__)

class MCPServer:
    """A JSON-RPC 2.0 server for MCP tool management with support for code and agent-based tools."""

    def __init__(self, config_path: str = "config/config.yaml", host: str = "0.0.0.0", port: int = 6000,
                 engine: Any = None, verbose: bool = False, log_level: str = "INFO"):
        """
        Initialize the MCP server with configuration, tool registry, and optional VLLM engine.

        Args:
            config_path (str): Path to configuration file.
            host (str): Host address (default: "0.0.0.0").
            port (int): Port to run the server on (default: 6000).
            engine (VLLMEngine, optional): Engine for agent tool execution.
            verbose (bool): If True, log detailed server operations.
            log_level (str): Logging level (e.g., "DEBUG", "INFO").
        """
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.engine = engine
        self.verbose = verbose
        self.config = self._load_config(config_path)
        self.registry_file = self.config["registry"]["path"]
        self.handler_config = self.config.get("handlers", {})
        self.handlers = {name: meta["func"] for name, meta in HANDLER_REGISTRY.items()}
        self.interceptor = GenerationInterceptor(self, engine, verbose=verbose) if engine else None
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self._initialize_handlers()
        self._setup_routes()
        if self.verbose:
            logger.info("Initialized MCPServer on %s:%d with registry %s", host, port, self.registry_file)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if not config or "registry" not in config:
                raise ValueError("Config must include 'registry' section")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")

    def _initialize_handlers(self):
        """Register primitive handlers from HANDLER_REGISTRY if not already present."""
        existing_tools = self._load_registry()
        existing_names = {tool["name"] for tool in existing_tools}
        for name, meta in HANDLER_REGISTRY.items():
            if name not in existing_names:
                tool_entry = {
                    "name": meta["name"],
                    "description": meta["description"],
                    "input": meta["input"],
                    "output": meta["output"],
                    "type": meta["type"],
                    "handler": meta["handler"],
                    "config": self.handler_config.get(name, {})  # Merge with config defaults
                }
                existing_tools.append(tool_entry)
                if self.verbose:
                    logger.info("Initialized primitive handler: %s", tool_entry)
        self._save_registry(existing_tools)

    def _setup_routes(self):
        """Define JSON-RPC 2.0 endpoints."""
        @self.app.route("/", methods=["POST"])
        def handle_jsonrpc():
            try:
                data = request.get_json()
                logger.debug("Received Payload: %s", data)
                if not data or "jsonrpc" not in data or data["jsonrpc"] != "2.0":
                    return jsonify({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": None}), 400

                method = data.get("method")
                params = data.get("params", {})
                request_id = data.get("id")

                if method == "list_tools":
                    result = self.list_tools()
                elif method == "register_tool":
                    result = self.register_tool(params)
                elif method == "invoke_tool":
                    tool_name = params.get("tool_name")
                    if not tool_name:
                        return jsonify({"jsonrpc": "2.0", "error": {"code": -32602, "message": "Missing tool_name"}, "id": request_id}), 400
                    if not params.get("consent", False):
                        return jsonify({"jsonrpc": "2.0", "error": {"code": -32500, "message": "Consent required"}, "id": request_id}), 400
                    result = self.invoke_tool(tool_name, params)
                else:
                    tools = self.list_tools()
                    if any(t["name"] == method for t in tools):
                        if not params.get("consent", False):
                            return jsonify({"jsonrpc": "2.0", "error": {"code": -32500, "message": "Consent required"}, "id": request_id}), 400
                        result = self.invoke_tool(method, params)
                    else:
                        return jsonify({"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": request_id}), 404

                return jsonify({"jsonrpc": "2.0", "result": result, "id": request_id})
            except Exception as e:
                logger.error("Error handling request: %s", str(e), exc_info=True)
                return jsonify({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": request_id}), 500

    def list_tools(self) -> List[Dict]:
        """Return the list of registered tools."""
        tools = self._load_registry()
        if self.verbose:
            logger.debug("Listing tools: %s", tools)
        return tools

    def register_tool(self, definition: Dict) -> Dict:
        """Register a new tool in the persistent registry."""
        if not isinstance(definition, dict) or "name" not in definition:
            raise ValueError("Definition must be a dictionary with a 'name' key")

        tool_entry = {
            "name": definition["name"],
            "description": definition.get("description", f"Tool {definition['name']}"),
            "input": definition.get("input", {}),
            "output": definition.get("output", "string"),
            "type": definition.get("type", "code")
        }

        if tool_entry["type"] == "code":
            if "handler" not in definition or definition["handler"] not in self.handlers:
                raise ValueError(f"Unknown or missing handler '{definition.get('handler', 'missing')}' for code-based tool")
            tool_entry["handler"] = definition["handler"]
            tool_entry["config"] = definition.get("config", {})
        elif tool_entry["type"] == "agent" and not self.engine:
            raise ValueError("Agent-based tools require a VLLMEngine instance")

        if "output_format" in definition:
            tool_entry["output_format"] = definition["output_format"]

        tools = self._load_registry()
        tools = [t for t in tools if t["name"] != tool_entry["name"]]
        tools.append(tool_entry)
        self._save_registry(tools)
        if self.verbose:
            logger.info("Registered tool: %s", tool_entry)
        return {"status": "success", "message": f"Tool '{tool_entry['name']}' registered"}

    def invoke_tool(self, tool_name: str, params: Dict) -> Dict:
        """Invoke a registered tool, handling both code-based and agent-based tools."""
        tools = self._load_registry()
        tool = next((t for t in tools if t["name"] == tool_name), None)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        if self.verbose:
            logger.info("Invoking tool '%s' with params: %s", tool_name, params)

        try:
            if tool["type"] == "agent" and self.engine and self.interceptor:
                template_info = load_prompt_template(tool_name, verbose=self.verbose)
                prompt = format_prompt(template_info, task=params.get("task", ""), tools=tools)
                generator = self.engine.generate_with_interception(prompt, self.interceptor, max_tokens=200)
                result = "".join(list(generator)).strip()
                parsed = parse_agent_output(result, tool_name, self.engine, self.verbose)
                if not parsed:
                    raise ValueError(f"Agent '{tool_name}' failed to produce valid output: {result}")
                return parsed

            elif tool["type"] == "code":
                handler_name = tool.get("handler")
                handler_config = {**self.handler_config.get(handler_name, {}), **tool.get("config", {})}
                handler_func = self.handlers.get(handler_name)
                if not handler_func:
                    return {"error": f"Handler '{handler_name}' not found for tool '{tool_name}'"}
                result = handler_func(handler_config, params)
                return {"result": result}

            else:
                return {"error": f"Unsupported tool type '{tool['type']}' for '{tool_name}'"}

        except Exception as e:
            error_msg = f"Failed to invoke tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

    def _load_registry(self) -> List[Dict]:
        """Load tools from the registry file with concurrency safety."""
        if not os.path.exists(self.registry_file):
            return []
        with open(self.registry_file, "r", encoding="utf-8") as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            try:
                tools = [json.loads(line) for line in f if line.strip()]
            finally:
                portalocker.unlock(f)
        return tools

    def _save_registry(self, tools: List[Dict]):
        """Save tools to the registry file with concurrency safety."""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                for tool in tools:
                    json.dump(tool, f)
                    f.write("\n")
            finally:
                portalocker.unlock(f)

    def run(self):
        """Start the MCP server."""
        try:
            if self.verbose:
                logger.info("Starting MCP server on %s:%d", self.host, self.port)
            self.app.run(host=self.host, port=self.port, debug=False)
        except Exception as e:
            logger.error("Failed to start server: %s", str(e), exc_info=True)
            raise RuntimeError(f"Server failed: {e}")

if __name__ == "__main__":
    server = MCPServer(verbose=True, debug=True, log_level="DEBUG")
    server.run()
