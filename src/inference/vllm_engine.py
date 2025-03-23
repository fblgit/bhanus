# ./src/inference/vllm_engine.py
import json
import logging
import os
from typing import Generator, List, Optional, Dict, Any
import yaml

from vllm import LLM, SamplingParams
#from src.generation.interceptor import GenerationInterceptor
#from src.mcp_client.client import MCPClient

# Configure logging
logger = logging.getLogger(__name__)

class VLLMEngine:
    """Manages text generation using the vLLM inference engine with function registration."""

    def __init__(self, config_path: str = "config/config.yaml", mcp_client: Any = None, verbose: bool = False, log_level: str = "INFO"):
        """
        Initialize the VLLMEngine with configuration and optional MCP client.

        Args:
            config_path (str): Path to the configuration file (default: "config/config.yaml").
            mcp_client (MCPClient, optional): Client for server-side function registration.
            verbose (bool): If True, log detailed generation steps.
            log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING").
        """
        self.llm: Optional[LLM] = None
        self.mcp_client = mcp_client
        self.verbose = verbose
        self.config = self._load_config(config_path)
        self.function_registry_file = self.config.get("registry", {}).get("path", "function_registry.json")
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                           format="%(asctime)s - %(levelname)s - %(message)s")
        if self.verbose:
            logger.debug("Initialized VLLMEngine with config: %s, MCP client: %s", self.config, "active" if mcp_client else "None")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from the specified YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict) or "vllm" not in config:
                raise ValueError("Invalid config: must contain 'vllm' section")
            return config
        except Exception as e:
            logger.error("Failed to load config from %s: %s", config_path, str(e))
            raise ValueError(f"Failed to load config: {e}")

    def load_model(self, model_name: str = None, **kwargs) -> None:
        """
        Load the specified vLLM model, defaulting to config if not provided.

        Args:
            model_name (str, optional): Name of the model (e.g., "Qwen/Qwen2.5-1.5B-Instruct").
            **kwargs: Additional arguments for vLLM LLM initialization (e.g., tensor_parallel_size).
        """
        model_name = model_name or self.config["vllm"].get("model", "Qwen/Qwen2.5-1.5B-Instruct")
        if not model_name or not isinstance(model_name, str):
            logger.error("Invalid model_name provided")
            raise ValueError("model_name must be a non-empty string")

        try:
            self.llm = LLM(model=model_name, **kwargs)
            if self.verbose:
                logger.info("Loaded model: %s with kwargs: %s", model_name, kwargs)
        except Exception as e:
            error_msg = f"Failed to load model '{model_name}' with kwargs {kwargs}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def register_function(self, name: str, definition: Dict) -> None:
        """
        Register or update a function created by the model with local and server-side persistence.

        Args:
            name (str): Name of the function (e.g., "tools_picker").
            definition (Dict): Function details (e.g., {"description": "...", "input": {}, "output": "...", "type": "..."}).
        """
        if not name or not isinstance(name, str):
            raise ValueError("Function name must be a non-empty string")
        if not isinstance(definition, dict):
            raise ValueError("Function definition must be a dictionary")

        func_entry = {
            "name": name,
            "description": definition.get("description", f"Function {name} created by model"),
            "input": definition.get("input", {}),
            "output": definition.get("output", "string"),
            "type": definition.get("type", "code")  # "code" or "agent"
        }
        if "output_format" in definition:
            func_entry["output_format"] = definition["output_format"]

        try:
            # Sync with MCP server if client is available
            if self.mcp_client:
                self.mcp_client.register_tool(name, func_entry)
                if self.verbose:
                    logger.info("Registered function '%s' with MCP server", name)

            # Update local registry
            existing_functions = self.list_registered_functions()
            updated = False
            for i, func in enumerate(existing_functions):
                if func["name"] == name:
                    existing_functions[i] = func_entry  # Update existing entry
                    updated = True
                    break
            if not updated:
                existing_functions.append(func_entry)  # Add new entry if not found

            with open(self.function_registry_file, "w", encoding="utf-8") as f:
                for entry in existing_functions:
                    json.dump(entry, f)
                    f.write("\n")
            if self.verbose:
                logger.info("Registered/updated function locally: %s with definition: %s", name, func_entry)

        except Exception as e:
            error_msg = f"Failed to register function '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def list_registered_functions(self) -> List[Dict]:
        """
        Retrieve all registered functions from the persistent store.

        Returns:
            List[Dict]: List of registered function definitions.
        """
        functions = []
        try:
            if not os.path.exists(self.function_registry_file):
                return functions
            with open(self.function_registry_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        functions.append(json.loads(line))
            if self.verbose:
                logger.debug("Listed registered functions: %s", functions)
            return functions
        except Exception as e:
            error_msg = f"Failed to list registered functions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def cleanup_registry(self) -> None:
        """
        Remove stale or invalid entries from the local function registry.
        """
        try:
            functions = self.list_registered_functions()
            valid_functions = []
            required_keys = {"name", "description", "input", "output", "type"}

            for func in functions:
                if not isinstance(func, dict) or not required_keys.issubset(func.keys()):
                    logger.warning("Removing invalid function entry: %s", func)
                    continue
                valid_functions.append(func)

            if len(valid_functions) < len(functions):
                with open(self.function_registry_file, "w", encoding="utf-8") as f:
                    for entry in valid_functions:
                        json.dump(entry, f)
                        f.write("\n")
                if self.verbose:
                    logger.info("Cleaned up registry; kept %d valid functions", len(valid_functions))
            elif self.verbose:
                logger.debug("No cleanup needed; all %d functions valid", len(functions))

        except Exception as e:
            error_msg = f"Failed to clean up registry: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def generate(self, prompt: str, stop_tokens: List[str] = None, **kwargs) -> Generator[str, None, None]:
        """
        Generate text based on the prompt, yielding complete output chunks.

        Args:
            prompt (str): Input prompt for generation.
            stop_tokens (List[str], optional): Tokens that stop generation.
            **kwargs: Additional parameters for SamplingParams (e.g., temperature, max_tokens).
        """
        if self.llm is None:
            error_msg = f"Model not loaded; attempted prompt: '{prompt}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not prompt or not isinstance(prompt, str):
            logger.error("Invalid prompt provided")
            raise ValueError("prompt must be a non-empty string")

        try:
            if self.verbose:
                logger.info("Generating with prompt: '%s'", prompt)
                logger.debug("Stop tokens: %s, kwargs: %s", stop_tokens, kwargs)

            default_stop_tokens = self.config.get("vllm", {}).get("stop_tokens", ["[[", "]]", "<function_call>", "</function_call>"])
            combined_stop_tokens = list(set((stop_tokens or []) + default_stop_tokens))
            sampling_params = SamplingParams(
                stop=combined_stop_tokens,
                include_stop_str_in_output=True,
                **kwargs
            )
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)

            for output in outputs:
                chunk = output.outputs[0].text
                if self.verbose:
                    logger.debug("Yielding chunk: '%s'", chunk)
                yield chunk

        except Exception as e:
            error_msg = f"Generation failed for prompt '{prompt}' with stop_tokens {stop_tokens}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def generate_with_interception(self, prompt: str, interceptor: Any, **kwargs) -> Generator[str, None, None]:
        """
        Generate text with interception for function calls.

        Args:
            prompt (str): Input prompt for generation.
            interceptor (GenerationInterceptor): Instance to process the stream.
            **kwargs: Additional parameters for generate.
        """
        #if not isinstance(interceptor, GenerationInterceptor):
        #    error_msg = "Invalid interceptor; must be an instance of GenerationInterceptor"
        #    logger.error(error_msg)
        #    raise ValueError(error_msg)

        try:
            if self.verbose:
                logger.info("Generating with interception for prompt: '%s'", prompt)
            generator = self.generate(prompt, **kwargs)
            intercepted_stream = interceptor.intercept_stream(generator)
            for chunk in intercepted_stream:
                if self.verbose:
                    logger.debug("Yielding intercepted chunk: '%s'", chunk)
                yield chunk

        except Exception as e:
            error_msg = f"Generation with interception failed for prompt '{prompt}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

# Example usage for testing
if __name__ == "__main__":
    class MockMCPClient:
        def register_tool(self, name: str, definition: Dict) -> None:
            logger.info("Mock: Registered %s with %s", name, definition)

    # Initialize engine
    client = MockMCPClient()
    engine = VLLMEngine(verbose=True, mcp_client=client, log_level="DEBUG")
    engine.load_model()

    # Test function registration
    print("\nTesting function registration:")
    engine.register_function("tools_picker", {
        "description": "Selects tools for a given task",
        "input": {"task": "string"},
        "output": "list",
        "type": "agent"
    })
    engine.register_function("tools_picker", {
        "description": "Updated tool selector",
        "input": {"task": "string", "context": "string"},
        "output": "list",
        "type": "agent",
        "output_format": "Selected tools: \\[.*\\]"
    })
    registered = engine.list_registered_functions()
    print("Registered functions:", registered)

    # Test cleanup
    print("\nTesting registry cleanup:")
    with open(engine.function_registry_file, "a", encoding="utf-8") as f:
        f.write("invalid_entry\n")
    engine.cleanup_registry()
    print("After cleanup:", engine.list_registered_functions())

    # Test basic generation
    prompt = "Define a function: [[define_function(tools_picker) Selects tools for a task]]"
    print("\nTesting generate:")
    for chunk in engine.generate(prompt, max_tokens=100):
        print(chunk, end="", flush=True)
