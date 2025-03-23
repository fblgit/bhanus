## `toolbox_catalog.md` Technical Specification

Below is the Markdown-style technical specification, including few-shot examples to reinforce the syntax for the agent system.

```markdown
# Toolbox Catalog Template Specification

## Overview
The `toolbox_catalog.jinja2` template generates a structured list of available tools in the system, formatted as a JSON-like string. This catalog is useful for debugging, tool discovery, or providing context to other prompts (e.g., `tools_picker`).

## Template Usage
- **Template File**: `toolbox_catalog.jinja2`
- **Inputs**:
  - `tools`: A list of dictionaries, where each dictionary represents a tool with the following keys:
    - `name` (str): The tool’s name (e.g., "get_time").
    - `description` (str): A brief description (e.g., "Gets current time").
    - `input` (dict): Expected input parameters (e.g., `{}` or `{"query": "string"}`).
    - `output` (str): The output type (e.g., "string", "dict").
    - `type` (str): The tool type ("code" or "agent").
- **Expected Output**: A string starting with "Here is the toolbox catalog: " followed by a JSON representation of the `tools` list.

## Syntax and Formatting
- The output must start with `"Here is the toolbox catalog: "`.
- The tools list must be a valid JSON structure enclosed in curly braces `{}`, with a key `"tools"` containing the array of tool dictionaries.
- Each tool dictionary must include all required keys: `name`, `description`, `input`, `output`, and `type`.

## Few-Shot Examples

### Example 1: Single Tool
- **Input**:
  ```python
  tools = [
      {"name": "get_time", "description": "Gets current time", "input": {}, "output": "string", "type": "code"}
  ]
  ```
- **Output**:
  ```
  Here is the toolbox catalog: {"tools": [{"name": "get_time", "description": "Gets current time", "input": {}, "output": "string", "type": "code"}]}
  ```

### Example 2: Multiple Tools
- **Input**:
  ```python
  tools = [
      {"name": "search_from_api", "description": "Search API", "input": {"query": "string"}, "output": "dict", "type": "code"},
      {"name": "tools_picker", "description": "Selects tools for a task", "input": {"task": "string"}, "output": "list", "type": "agent"}
  ]
  ```
- **Output**:
  ```
  Here is the toolbox catalog: {"tools": [{"name": "search_from_api", "description": "Search API", "input": {"query": "string"}, "output": "dict", "type": "code"}, {"name": "tools_picker", "description": "Selects tools for a task", "input": {"task": "string"}, "output": "list", "type": "agent"}]}
  ```

### Example 3: Empty Catalog
- **Input**:
  ```python
  tools = []
  ```
- **Output**:
  ```
  Here is the toolbox catalog: {"tools": []}
  ```

## Validation
- The output is validated against the regex pattern `Here is the toolbox catalog: \{.*\}` to ensure it starts correctly and contains a JSON-like structure.
- Each tool dictionary in the JSON must contain all required keys (`name`, `description`, `input`, `output`, `type`).

## Notes
- The `tojson` filter in Jinja2 ensures proper JSON formatting of the `tools` list.
- Since the syntax isn’t native to the models, the agent must strictly follow the structure shown in the examples to ensure compatibility with downstream systems (e.g., `validator.py`).
