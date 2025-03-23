import logging
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, TemplateError, UndefinedError

logger = logging.getLogger(__name__)

def load_prompt_template(template_name: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Load a Jinja2 template and extract metadata from its comments.

    Args:
        template_name (str): Name of the template file (e.g., "define_function").
        verbose (bool): If True, log detailed loading steps.

    Returns:
        Dict[str, Any]: Dictionary with 'template', 'expected_format', and 'description' keys.

    Raises:
        FileNotFoundError: If the template file is not found.
        ValueError: If the template is malformed or loading fails.
    """
    try:
        # Set up Jinja2 environment
        template_dir = "templates"  # Adjust this path based on your project structure
        environment = Environment(loader=FileSystemLoader(template_dir))
        if not template_name.endswith(".jinja2"):
            template_name += ".jinja2"

        # Load the template
        template = environment.get_template(template_name)
        if verbose:
            logger.debug(f"Loaded Jinja2 template: {template_name}")

        # Fetch the raw template source using the loader
        template_source, _, _ = environment.loader.get_source(environment, template_name)
        if verbose:
            logger.debug(f"Raw template source:\n{template_source}")

        # Parse metadata from comments
        expected_format = ".*"  # Default
        description = "Generic template"
        for line in template_source.splitlines():
            if line.strip().startswith("{#"):
                comment = line.strip()[2:-2].strip()
                if "expected_format:" in comment:
                    expected_format = comment.split("expected_format:")[1].strip()
                elif "description:" in comment:
                    description = comment.split("description:")[1].strip()

        template_info = {
            "template": template,
            "expected_format": expected_format,
            "description": description
        }
        if verbose:
            logger.info(f"Loaded template '{template_name}' with expected_format '{expected_format}' and description '{description}'")
        return template_info

    except TemplateNotFound as e:
        error_msg = f"Template file not found: {template_name}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except TemplateError as e:
        error_msg = f"Malformed template '{template_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Failed to load template '{template_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)

def format_prompt(template_info: Dict[str, Any], validate: bool = True, verbose: bool = False, **kwargs: Any) -> str:
    """
    Render a Jinja2 template with provided values and optionally validate the output.

    Args:
        template_info (Dict[str, Any]): Loaded template info with 'template', 'expected_format', and 'description' keys.
        validate (bool): If True, validate the rendered output against the expected format.
        verbose (bool): If True, log detailed rendering steps.
        **kwargs: Values to substitute into the template (e.g., tools=[...], task="...").

    Returns:
        str: Rendered and validated prompt string.

    Raises:
        KeyError: If a required variable is missing, with specific details.
        ValueError: If rendering or validation fails, with detailed context.
    """
    if not isinstance(template_info, dict) or "template" not in template_info:
        error_msg = "Invalid template_info: must be a dictionary with 'template' key"
        logger.error(error_msg)
        raise ValueError(error_msg)

    template = template_info["template"]
    expected_format = template_info.get("expected_format", ".*")
    description = template_info.get("description", "Unknown template")

    try:
        rendered = template.render(**kwargs)
        if verbose:
            logger.debug(f"Rendered prompt for '{description}': '{rendered}' with kwargs: {kwargs}")

        if validate:
            from .validator import PromptValidator  # Moved import here to avoid circular dependency
            validator = PromptValidator(verbose=verbose)
            if not validator.validate(rendered, expected_format, template_name=description):
                error_msg = f"Rendered prompt for '{description}' does not match expected format '{expected_format}': '{rendered}'"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if verbose:
                logger.info(f"Validated prompt for '{description}': '{rendered}'")

        return rendered

    except UndefinedError as e:
        # Handle missing variables specifically
        missing_var = str(e).split("'")[1] if "'" in str(e) else str(e)
        error_msg = f"Missing required variable '{missing_var}' in template '{description}' with kwargs {kwargs}"
        logger.error(error_msg)
        raise KeyError(error_msg)
    except TemplateError as e:
        error_msg = f"Failed to render template '{description}' with kwargs {kwargs}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error rendering template '{description}' with kwargs {kwargs}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)

# Example usage for testing
if __name__ == "__main__":
    # Create a mock template for testing
    from jinja2 import Template
    mock_template_info = {
        "template": Template("Task: {{ task }}\nTools: {{ tools | tojson }} {# expected_format: Task: .+\\nTools: .* #} {# description: Mock tools picker #}"),
        "expected_format": "Task: .+\\nTools: .*",
        "description": "Mock tools picker"
    }
    verbose = True

    try:
        # Test successful rendering and validation
        tools = [{"name": "get_time", "description": "Gets current time", "input": {}, "output": "string", "type": "code"}]
        prompt = format_prompt(mock_template_info, task="summarize code", tools=tools, verbose=verbose)
        print(f"Rendered Prompt:\n{prompt}")

        # Test missing variable
        prompt = format_prompt(mock_template_info, tools=tools, verbose=verbose)  # Missing 'task'
    except Exception as e:
        print(f"Error: {e}")

    try:
        # Test invalid output against expected format
        invalid_template_info = {
            "template": Template("Invalid output"),
            "expected_format": "Task: .+",
            "description": "Invalid template"
        }
        prompt = format_prompt(invalid_template_info, validate=True, verbose=verbose)
    except Exception as e:
        print(f"Error: {e}")
