# ./src/generation/parser.py
import json
import logging
import re
from typing import Optional, Dict, Any

#from src.inference.vllm_engine import VLLMEngine  # Added for registry access

# Configure logging
logger = logging.getLogger(__name__)

def extract_function_call(text: str, verbose: bool = False) -> Optional[Dict]:
    """
    Extract function call or definition details from text, supporting multiple syntaxes.

    Args:
        text (str): Text containing a potential function call or definition.
        verbose (bool): If True, log detailed parsing steps.

    Returns:
        Optional[Dict]: Dictionary with 'name' (str) and 'params' (Dict) if found, else None.
    """
    try:
        if verbose:
            logger.debug("Parsing text for function call or definition: '%s'", text)

        formats = [
            {
                "start": r"\[\[",
                "end": r"\]\]",
                "extract": lambda t: re.search(r"\[\[(.*?)\]\]", t)
            },
            {
                "start": r"<function_call>",
                "end": r"</function_call>",
                "extract": lambda t: re.search(r"<function_call>(.*?)</function_call>", t)
            }
        ]

        for fmt in formats:
            match = fmt["extract"](text)
            if match:
                call_str = match.group(1)
                if verbose:
                    logger.info("Found potential function call/definition: '%s' in format %s...%s", call_str, fmt["start"], fmt["end"])

                if fmt["start"] == r"\[\[" and "define_function(" in call_str:
                    try:
                        def_match = re.match(r"define_function\((.*?)\)\s*(.*)", call_str)
                        if not def_match:
                            logger.warning("Invalid define_function syntax: '%s'", call_str)
                            return None
                        name, description = def_match.groups()
                        definition = {
                            "description": description.strip(),
                            "input": {},
                            "output": "string",
                            "type": "agent" if "agent" in description.lower() else "code"
                        }
                        result = {
                            "name": "define_function",
                            "params": {"name": name.strip(), "definition": definition}
                        }
                        if verbose:
                            logger.info("Parsed function definition: %s", result)
                        return result
                    except ValueError as e:
                        logger.warning("Failed to parse define_function '%s': %s", call_str, e)
                        return None

                if fmt["start"] == r"\[\[":
                    try:
                        tool_name, params_str = call_str.split("(", 1)
                        params_str = params_str.rstrip(")")
                        tool_name = tool_name.strip()
                        if not tool_name:
                            logger.warning("Tool name is empty")
                            return None

                        params = {}
                        if params_str.strip():
                            param_pairs = re.findall(r'(\w+)=("[^"]*"|[^,\s"]+)(?:,|$)', params_str)
                            if not param_pairs and params_str:
                                logger.warning("Failed to parse parameters: '%s'", params_str)
                                return None
                            for key, value in param_pairs:
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                params[key] = value
                            if verbose:
                                logger.debug("Parsed parameters: %s", params)

                        result = {"name": tool_name, "params": params}
                        if verbose:
                            logger.info("Parsed bracket function call: %s", result)
                        return result
                    except ValueError as e:
                        logger.warning("Invalid bracket function call syntax: '%s' - %s", call_str, e)
                        return None

                if fmt["start"] == r"<function_call>":
                    try:
                        call_dict = json.loads(call_str)
                        if not isinstance(call_dict, dict) or "name" not in call_dict:
                            logger.warning("Invalid JSON function call structure: '%s'", call_str)
                            return None
                        result = {
                            "name": call_dict["name"],
                            "params": call_dict.get("params", {})
                        }
                        if verbose:
                            logger.info("Parsed JSON function call: %s", result)
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSON function call '%s': %s", call_str, e)
                        return None

        if verbose:
            logger.debug("No function call or definition found in text")
        return None

    except Exception as e:
        logger.error("Error parsing function call in text '%s': %s", text, str(e), exc_info=True)
        return None

def parse_agent_output(text: str, agent_name: str, engine: Any = None, verbose: bool = False) -> Optional[Dict]:
    """
    Parse the output of an agent-based function dynamically, supporting Jinja2-rendered formats.

    Args:
        text (str): Output text from the agent.
        agent_name (str): Name of the agent (e.g., "tools_picker").
        engine (VLLMEngine, optional): Engine instance to access registered functions for dynamic parsing.
        verbose (bool): If True, log detailed parsing steps.

    Returns:
        Optional[Dict]: Parsed result (e.g., {"selected_tools": [...]}) if valid, else None.
    """
    try:
        if verbose:
            logger.debug("Parsing agent output: '%s' for agent '%s'", text, agent_name)

        from src.prompts.validator import PromptValidator
        validator = PromptValidator(verbose=verbose)
        
        # Load agent definition from registry if engine is provided
        expected_key = None
        expected_format = ".*"
        if engine:
            registered_functions = engine.list_registered_functions()
            agent_def = next((f for f in registered_functions if f["name"] == agent_name and f.get("type") == "agent"), None)
            if agent_def:
                expected_key = "selected_tools" if agent_name == "tools_picker" else "agents" if agent_name == "agent_factory" else "result"
                expected_format = agent_def.get("output_format", f"{expected_key.capitalize()}: \\[.*\\]")
                if verbose:
                    logger.debug("Loaded agent definition for '%s': expected_key=%s, format=%s", agent_name, expected_key, expected_format)

        # Standardize to JSON if possible
        if isinstance(text, dict):
            try:
                data = text
                if expected_key and expected_key in data and isinstance(data[expected_key], list):
                    if verbose:
                        logger.info("Parsed JSON agent output: %s", data)
                    return data
                elif "result" in data:
                    if verbose:
                        logger.info("Parsed generic JSON result: %s", data)
                    return data
                logger.warning("Invalid JSON structure for agent '%s': %s", agent_name, text)
            except json.JSONDecodeError:
                logger.debug("Text starts with '{' but isn't valid JSON: '%s'", text)

        # Standardize to JSON if possible
        elif isinstance(text, str) and text.strip().startswith("{"):
            try:
                data = json.loads(text)
                if expected_key and expected_key in data and isinstance(data[expected_key], list):
                    if verbose:
                        logger.info("Parsed JSON agent output: %s", data)
                    return data
                elif "result" in data:
                    if verbose:
                        logger.info("Parsed generic JSON result: %s", data)
                    return data
                logger.warning("Invalid JSON structure for agent '%s': %s", agent_name, text)
            except json.JSONDecodeError:
                logger.debug("Text starts with '{' but isn't valid JSON: '%s'", text)

        # Regex-based parsing for Jinja2-rendered output
        pattern = rf"{expected_key.capitalize()}: \[([^\]]+)\]" if expected_key else r"(\w+): \[([^\]]+)\]"
        match = re.search(pattern, text.splitlines()[-1] if "\n" in text else text)
        if match:
            if expected_key:
                items = [item.strip() for item in match.group(1).split(",") if item.strip()]
                result = {expected_key: items}
                if validator.validate(text, expected_format):
                    if verbose:
                        logger.info("Parsed agent output with regex: %s", result)
                    return result
            else:
                key, items_str = match.groups()
                items = [item.strip() for item in items_str.split(",") if item.strip()]
                result = {key.lower(): items}
                if verbose:
                    logger.info("Parsed generic key-value output: %s", result)
                return result

        # Broader recovery attempts
        if "[" in text and "]" in text:
            try:
                list_content = text[text.index("[") + 1:text.index("]")]
                items = [item.strip() for item in list_content.split(",") if item.strip()]
                result = {expected_key or "result": items}
                if verbose:
                    logger.debug("Recovered list from text: %s", result)
                return result
            except Exception as e:
                logger.debug("List recovery failed: %s", e)

        # Key-value pair recovery
        kv_pairs = re.findall(r"(\w+):\s*([^\n]+)", text)
        if kv_pairs:
            result = {key.lower(): value.strip() for key, value in kv_pairs}
            if expected_key and expected_key in result:
                if verbose:
                    logger.info("Recovered key-value pairs: %s", result)
                return result

        logger.warning("Failed to parse agent output for '%s': no valid format detected in '%s'", agent_name, text)
        return None

    except Exception as e:
        logger.error("Error parsing agent output '%s' for '%s': %s", text, agent_name, str(e), exc_info=True)
        return None

# Example usage for testing
if __name__ == "__main__":
    class MockVLLMEngine:
        def list_registered_functions(self):
            return [
                {"name": "tools_picker", "type": "agent", "output_format": "Selected tools: \\[.*\\]"},
                {"name": "agent_factory", "type": "agent", "output_format": "Agents: \\[.*\\]"}
            ]

    engine = MockVLLMEngine()
    test_cases = [
        # Function call tests
        ("Hello [[search_from_api(query=\"AI news\", limit=5)]] world", None),
        ("[[define_function(my_tool) Custom tool description]]", None),
        ("<function_call>{\"name\": \"get_current_time\", \"params\": {}}</function_call>", None),
        # Agent output tests
        ("Selected tools: [parse_html, summarise_text]", "tools_picker"),
        ("Agents: [scraper, summarizer]", "agent_factory"),
        ("{\"selected_tools\": [\"parse_html\", \"summarise_text\"]}", "tools_picker"),
        ("Invalid output with [random, stuff]", "tools_picker"),
        ("Key: value\nTools: parse_html, summarise_text", "tools_picker"),
        ("Mixed format [tool1] extra text", "tools_picker")
    ]

    print("Testing extract_function_call:")
    for test, agent in test_cases[:3]:
        print(f"\nTesting: '{test}'")
        result = extract_function_call(test, verbose=True)
        print(f"Result: {result}")

    print("\nTesting parse_agent_output:")
    for test, agent in test_cases[3:]:
        print(f"\nTesting: '{test}' for agent '{agent}'")
        result = parse_agent_output(test, agent, engine=engine, verbose=True)
        print(f"Result: {result}")
