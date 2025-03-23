# ./src/prompts/validator.py
import json
import logging
import re
from typing import Optional, Dict, List, Any

from src.generation.parser import extract_function_call
from src.prompts.utils import load_prompt_template
#from src.inference.vllm_engine import VLLMEngine  # Added for registry access

# Configure logging
logger = logging.getLogger(__name__)

class PromptValidator:
    """Validates model output against expected formats from Jinja2 templates or registry."""

    def __init__(self, engine: Any = None, verbose: bool = False, log_level: str = "INFO"):
        """
        Initialize the PromptValidator with optional engine for registry access.

        Args:
            engine (VLLMEngine, optional): Engine instance to access registered functions.
            verbose (bool): If True, log detailed validation steps.
            log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING").
        """
        self.engine = engine
        self.verbose = verbose
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                           format="%(asctime)s - %(levelname)s - %(message)s")
        if self.verbose:
            logger.debug("Initialized PromptValidator with log level %s and %s engine", log_level, "active" if engine else "no")

    def validate(self, output: str, expected_format: str, template_name: str = None) -> bool:
        """
        Validate the model's output against the expected format, using template or registry metadata.

        Args:
            output (str): The output from the model to validate.
            expected_format (str): The expected format pattern (e.g., ".*", "Selected tools: \\[.*\\]").
            template_name (str, optional): Name of the Jinja2 template or agent for context.

        Returns:
            bool: True if the output matches the expected format, False otherwise.
        """
        if not output or not expected_format:
            logger.warning("Output or expected_format is empty")
            return False

        if self.verbose:
            logger.debug("Validating output: '%s' against format: '%s' with template: %s", output, expected_format, template_name or "None")

        try:
            # Handle intermediate function calls or definitions
            call_dict = extract_function_call(output, verbose=self.verbose)
            if call_dict:
                if call_dict["name"] == "define_function":
                    if self.verbose:
                        logger.info("Validated function definition: %s", call_dict)
                    return True
                if self.verbose:
                    logger.info("Validated standard or agent function call: %s", call_dict)
                return True  # Function calls are valid intermediates

            # Handle registration confirmation
            if output.startswith("[registered]") and output.endswith("[/registered]"):
                if self.verbose:
                    logger.info("Validated registration confirmation: '%s'", output)
                return True

            # Handle error output
            if output.startswith("[error]") and output.endswith("[/error]"):
                logger.warning("Output is an error: '%s'", output)
                return False

            # Load template or registry metadata if provided
            if template_name and self.engine:
                registered_functions = self.engine.list_registered_functions()
                agent_def = next((f for f in registered_functions if f["name"] == template_name and f.get("type") == "agent"), None)
                if agent_def:
                    expected_format = agent_def.get("output_format", expected_format)
                    if self.verbose:
                        logger.debug("Updated expected_format from registry for '%s': '%s'", template_name, expected_format)
                elif template_name.endswith(".jinja2"):
                    try:
                        template_info = load_prompt_template(template_name[:-7], verbose=self.verbose)
                        expected_format = template_info.get("expected_format", expected_format)
                        if self.verbose:
                            logger.debug("Updated expected_format from template '%s': '%s'", template_name, expected_format)
                    except Exception as e:
                        logger.warning("Failed to load template '%s' for validation: %s", template_name, e)

            # Perform validation
            return self._validate_output(output, expected_format, template_name)

        except Exception as e:
            logger.error("Validation failed for output '%s' against '%s': %s", output, expected_format, e, exc_info=True)
            return False

    def _validate_output(self, output: str, expected_format: str, template_name: str = None) -> bool:
        """
        Validate output against expected formats dynamically.

        Args:
            output (str): The raw output from the model.
            expected_format (str): The expected format pattern (regex or descriptive).
            template_name (str, optional): Context for logging and dynamic validation.

        Returns:
            bool: True if the output matches the format, False otherwise.
        """
        if expected_format == ".*":
            return bool(output.strip())  # Non-empty output is valid for generic templates

        # Handle regex patterns
        if expected_format.startswith("\\") or expected_format.endswith("\\"):
            try:
                target_line = output.splitlines()[-1] if "\n" in output else output
                if re.match(expected_format, target_line):
                    if self.verbose:
                        logger.debug("Validated output against regex: '%s'", expected_format)
                    return True
                logger.warning("Output '%s' does not match regex format '%s'", output, expected_format)
                return False
            except re.error as e:
                logger.error("Invalid regex pattern '%s': %s", expected_format, e)
                return False

        # Generic JSON structure validation
        if output.strip().startswith("{"):
            try:
                data = json.loads(output)
                if isinstance(data, dict):
                    if template_name and template_name in ["tools_picker", "agent_factory"]:
                        key = "selected_tools" if template_name == "tools_picker" else "agents"
                        if key in data and isinstance(data[key], list):
                            if self.verbose:
                                logger.debug("Validated JSON structure for '%s': %s", template_name, data)
                            return True
                    elif "result" in data or any(isinstance(v, (list, str)) for v in data.values()):
                        if self.verbose:
                            logger.debug("Validated generic JSON structure: %s", data)
                        return True
                logger.warning("Invalid JSON structure for '%s': %s", template_name or "generic", output)
                return False
            except json.JSONDecodeError:
                logger.debug("Output starts with '{' but isn't valid JSON: '%s'", output)

        # Generic list validation
        if "[" in output and "]" in output:
            list_content = output[output.index("[") + 1:output.index("]")]
            items = [item.strip() for item in list_content.split(",") if item.strip()]
            if items:
                if self.verbose:
                    logger.debug("Validated list structure: %s", items)
                return True

        logger.warning("Unsupported or unmatched expected_format: '%s' for output: '%s' (template: %s)", 
                       expected_format, output, template_name or "None")
        return False

# Example usage for testing
if __name__ == "__main__":
    class MockVLLMEngine:
        def list_registered_functions(self):
            return [
                {"name": "tools_picker", "type": "agent", "output_format": "Selected tools: \\[.*\\]"},
                {"name": "agent_factory", "type": "agent", "output_format": "Agents: \\[.*\\]"},
                {"name": "custom_agent", "type": "agent", "output_format": "Results: \\[.*\\]"}
            ]

    engine = MockVLLMEngine()
    validator = PromptValidator(engine=engine, verbose=True, log_level="DEBUG")
    test_cases = [
        ("[[define_function(my_tool) Custom tool]]", ".*", None),
        ("[registered]my_tool[/registered]", ".*", None),
        ("[[tools_picker(task=\"summarize code\")]]", ".*", None),
        ("Selected tools: [parse_html, summarise_text]", "Selected tools: \\[.*\\]", "tools_picker"),
        ("Agents: [scraper, summarizer]", "Agents: \\[.*\\]", "agent_factory"),
        ("Results: [result1, result2]", "Results: \\[.*\\]", "custom_agent"),
        ("{\"selected_tools\": [\"parse_html\", \"summarise_text\"]}", ".*", "tools_picker"),
        ("[error]Something went wrong[/error]", ".*", None),
        ("Invalid output", "Selected tools: \\[.*\\]", "tools_picker")
    ]

    for output, expected_format, template_name in test_cases:
        print(f"\nValidating: '{output}' against '{expected_format}' with template '{template_name or 'None'}'")
        result = validator.validate(output, expected_format, template_name)
        print(f"Result: {result}")
