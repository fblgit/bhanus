# ./src/generation/interceptor.py
import json
import logging
from typing import Generator, Optional, Dict, List, Any

#from src.mcp_client.client import MCPClient
#from src.inference.vllm_engine import VLLMEngine
from src.generation.parser import extract_function_call, parse_agent_output
from src.prompts.utils import load_prompt_template, format_prompt

# Configure logging with dynamic level
logger = logging.getLogger(__name__)

class GenerationInterceptor:
    """Intercepts the generation stream to detect and handle function calls and definitions."""

    def __init__(self, mcp_client: Any, engine: Any, verbose: bool = False, log_level: str = "INFO"):
        """
        Initialize the GenerationInterceptor with MCP client and VLLM engine.

        Args:
            mcp_client (MCPClient): Client for invoking and registering tools on the MCP server.
            engine (VLLMEngine): Engine for recursive generation in agent calls.
            verbose (bool): If True, enable detailed logging.
            log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING").
        """
        self.mcp_client = mcp_client
        self.engine = engine
        self.verbose = verbose
        self.buffer = ""
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                           format="%(asctime)s - %(levelname)s - %(message)s")
        if self.verbose:
            logger.debug("Initialized GenerationInterceptor with MCPClient and VLLMEngine at level %s", log_level)

    def intercept_stream(self, generator: Generator[str, None, None]) -> Generator[str, None, None]:
        """
        Process the generation stream, handling function calls and definitions across chunks.

        Args:
            generator (Generator[str, None, None]): Stream of text chunks from VLLMEngine.

        Yields:
            str: Processed text chunks, including function call results or registration confirmations.
        """
        try:
            for chunk in generator:
                self.buffer += chunk
                if self.verbose:
                    logger.debug("Received chunk: '%s', Buffer: '%s'", chunk, self.buffer)

                while True:  # Process all complete calls in the buffer
                    call_dict = self._detect_function_call(self.buffer)
                    if call_dict:
                        start_idx = self.buffer.index(call_dict["start_marker"])
                        end_idx = self.buffer.index(call_dict["end_marker"]) + len(call_dict["end_marker"])

                        # Yield any text before the function call
                        if start_idx > 0:
                            pre_call_text = self.buffer[:start_idx]
                            if self.verbose:
                                logger.debug("Yielding pre-call text: '%s'", pre_call_text)
                            yield pre_call_text

                        # Process the function call
                        result = self._handle_function_call(call_dict)
                        if self.verbose:
                            logger.debug("Function call result: '%s'", result)
                        yield result

                        # Update buffer to remove processed portion
                        self.buffer = self.buffer[end_idx:]
                        if self.verbose:
                            logger.debug("Updated buffer after processing: '%s'", self.buffer)
                    elif self._has_partial_function_call(self.buffer):
                        if self.verbose:
                            logger.debug("Partial function call detected; awaiting more chunks")
                        break  # Wait for more chunks to complete the call
                    else:
                        # No function call or partial; yield buffer if complete, otherwise wait
                        if self.buffer:
                            if self.verbose:
                                logger.debug("No function call detected; yielding buffer: '%s'", self.buffer)
                            yield self.buffer
                            self.buffer = ""
                        break

            # Handle any remaining buffer at stream end
            if self.buffer:
                if self._has_partial_function_call(self.buffer):
                    logger.warning("Stream ended with incomplete function call: '%s'", self.buffer)
                    yield f"[error]Incomplete function call: {self.buffer}[/error]"
                else:
                    if self.verbose:
                        logger.debug("End of stream; yielding remaining buffer: '%s'", self.buffer)
                    yield self.buffer
                self.buffer = ""

        except Exception as e:
            error_msg = f"Error in intercept_stream with buffer '{self.buffer}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _detect_function_call(self, buffer: str) -> Optional[Dict]:
        """
        Detect a complete function call or definition in the buffer.

        Args:
            buffer (str): Accumulated text from the stream.

        Returns:
            Optional[Dict]: Dictionary with 'name', 'params', 'start_marker', 'end_marker' if found, else None.
        """
        formats = [
            {"start": "[[", "end": "]]", "parser": lambda x: extract_function_call(x, self.verbose)},
            {"start": "<function_call>", "end": "</function_call>", "parser": lambda x: json.loads(x)}
        ]

        if "[[define_function(" in buffer and "]]" in buffer:
            start_idx = buffer.index("[[define_function(")
            end_idx = buffer.index("]]") + len("]]")
            call_str = buffer[start_idx + 2:end_idx - 2]
            try:
                name_part, desc = call_str.split(")", 1)
                name = name_part.replace("define_function(", "").strip()
                definition = {
                    "description": desc.strip(),
                    "input": {},
                    "output": "string",
                    "type": "agent" if "agent" in desc.lower() else "code"
                }
                if self.verbose:
                    logger.info("Detected function definition: %s with %s", name, definition)
                return {"name": "define_function", "params": {"name": name, "definition": definition},
                        "start_marker": "[[", "end_marker": "]]"}
            except ValueError as e:
                logger.warning("Failed to parse function definition '%s': %s", call_str, e)
                return None

        for fmt in formats:
            start_marker = fmt["start"]
            end_marker = fmt["end"]
            if start_marker in buffer and end_marker in buffer:
                start_idx = buffer.index(start_marker)
                end_idx = buffer.index(end_marker) + len(end_marker)
                call_str = buffer[start_idx:end_idx]
                try:
                    parsed = fmt["parser"](call_str[len(start_marker):-len(end_marker)])
                    if parsed and isinstance(parsed, dict) and "name" in parsed:
                        if self.verbose:
                            logger.info("Detected function call: %s in format %s...%s", parsed, start_marker, end_marker)
                        return {"name": parsed["name"], "params": parsed.get("params", {}),
                                "start_marker": start_marker, "end_marker": end_marker}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse function call '%s': %s", call_str, e)
        return None

    def _has_partial_function_call(self, buffer: str) -> bool:
        """
        Check if the buffer contains the start of a function call or definition without its end.
        """
        starts = ["[[", "<function_call>"]
        ends = ["]]", "</function_call>"]
        for start, end in zip(starts, ends):
            if start in buffer and end not in buffer:
                return True
        return False

    def _handle_function_call(self, call_dict: Dict) -> str:
        """
        Execute a function call or register a new function definition via MCP client.

        Args:
            call_dict (Dict): Dictionary with 'name', 'params', 'start_marker', 'end_marker'.

        Returns:
            str: Result as a string for stream injection.
        """
        try:
            tool_name = call_dict["name"]
            params = call_dict["params"]

            if tool_name == "define_function":
                name = params["name"]
                definition = params["definition"]
                self.mcp_client.register_tool(name, definition)
                self.engine.register_function(name, definition)
                return f"[registered]{name}[/registered]"

            if tool_name in ["tools_picker", "agent_factory"]:
                return self._execute_agent_call(tool_name, params)

            if self.verbose:
                logger.info("Handling function call: %s with params: %s", tool_name, params)
            result = self.mcp_client.invoke_tool(tool_name, params, consent=True)
            formatted_result = json.dumps(result) if isinstance(result, dict) else str(result)
            if self.verbose:
                logger.info("Tool invocation successful: %s", formatted_result)
            return formatted_result

        except Exception as e:
            error_msg = f"Failed to handle function call '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"[error]{error_msg}[/error]"

    def _execute_agent_call(self, tool_name: str, params: Dict) -> str:
        """
        Execute an agent-based function by generating a response recursively with validation.

        Args:
            tool_name (str): Name of the agent tool (e.g., "tools_picker").
            params (Dict): Parameters for the agent.

        Returns:
            str: Validated result of the agent execution.
        """
        if self.verbose:
            logger.info("Executing agent call: %s with params: %s", tool_name, params)

        template_name = f"{tool_name}"
        try:
            template_info = load_prompt_template(template_name, verbose=self.verbose)
            tools = self.mcp_client.list_tools(include_local=True)
            prompt = format_prompt(template_info, task=params.get("task", ""), tools=tools)
        except Exception as e:
            logger.warning("Failed to load template for %s: %s; using fallback prompt", tool_name, e)
            prompt = f"{tool_name.capitalize()} task: '{params.get('task', '')}' with tools: {json.dumps(tools)}"

        generator = self.engine.generate_with_interception(prompt, self, max_tokens=200)
        result = "".join(list(generator)).strip()
        parsed_result = parse_agent_output(result, tool_name, verbose=self.verbose)

        if not parsed_result:
            error_msg = f"Agent '{tool_name}' returned invalid output: '{result}'"
            logger.error(error_msg)
            return f"[error]{error_msg}[/error]"

        formatted_result = json.dumps(parsed_result)
        if self.verbose:
            logger.debug("Agent '%s' validated result: %s", tool_name, formatted_result)
        return formatted_result

# Example usage for testing
if __name__ == "__main__":
    class MockMCPClient:
        def invoke_tool(self, tool_name: str, params: Dict, consent: bool) -> Dict:
            return {"result": f"Mock result for {tool_name} with {params}"}

        def register_tool(self, name: str, definition: Dict) -> None:
            logger.info("Mock: Registered %s with %s", name, definition)

        def list_tools(self, include_local: bool = False) -> List[Dict]:
            return [{"name": "get_time", "description": "Get time", "input": {}, "output": "string", "type": "code"}]

    class MockVLLMEngine:
        def register_function(self, name: str, definition: Dict) -> None:
            logger.info("Mock: Registered function %s with %s", name, definition)

        def generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
            yield prompt

        def generate_with_interception(self, prompt: str, interceptor, **kwargs) -> Generator[str, None, None]:
            yield from interceptor.intercept_stream(self.generate(prompt))

    engine = MockVLLMEngine()
    client = MockMCPClient()
    interceptor = GenerationInterceptor(client, engine, verbose=True, log_level="DEBUG")
    mock_generator = iter([
        "Hello [[define_function(my_tool) Custom tool]] world",
        "Text [[tools_picker(task=\"summarize code\"",
        "]] end",
        "Simple [[get_time()]] call"
    ])
    for output in interceptor.intercept_stream(mock_generator):
        print(output)
