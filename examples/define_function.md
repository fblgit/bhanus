# `define_function.md`
## Purpose
The `define_function.jinja2` template instructs the model to generate a function definition for registration in the MCP system. It supports both code-based and agent-based functions, allowing dynamic expansion of the system’s capabilities.

## Syntax
The expected syntax for defining a new function is:

```
[[define_function(function_name) function_description]]
```

- **function_name**: A unique string identifier for the function (e.g., "report_writer").
- **function_description**: A brief description of what the function does (e.g., "Writes a report from summarized content as an agent").

## Inputs
The template accepts the following inputs:
- `name` (str): The name of the function to define.
- `description` (str): A description of the function’s purpose.
- `input` (dict, optional): Expected input parameters (e.g., `{"summaries": "list"}`).
- `output` (str, optional): Expected output type (e.g., "string").
- `type` (str, optional): The function type, either "code" or "agent".

**Note**: While `input`, `output`, and `type` are optional in the basic syntax, they are important for fully specifying the function’s behavior in the system.

## Expected Output
The model should produce a string in the specified syntax, such as:

```
[[define_function(report_writer) Writes a report from summarized content as an agent]]
```

This string is then processed by the system to register the function.

## Few-Shot Examples
Since the syntax is not native to the models, these examples are critical for teaching the agent how to format its outputs correctly.

### Example 1: Code-Based Function
- **Inputs**:
  - `name`: "get_current_time"
  - `description`: "Retrieves the current system time"
  - `input`: `{}`
  - `output`: "string"
  - `type`: "code"
- **Expected Output**:
  ```
  [[define_function(get_current_time) Retrieves the current system time]]
  ```

### Example 2: Agent-Based Function
- **Inputs**:
  - `name`: "tools_picker"
  - `description`: "Selects relevant tools for a given task"
  - `input`: `{"task": "string"}`
  - `output`: "list"
  - `type`: "agent"
- **Expected Output**:
  ```
  [[define_function(tools_picker) Selects relevant tools for a given task]]
  ```

### Example 3: Function with Parameters
- **Inputs**:
  - `name`: "search_from_api"
  - `description`: "Searches an API for a query and returns results"
  - `input`: `{"query": "string", "limit": "int"}`
  - `output`: "dict"
  - `type`: "code"
- **Expected Output**:
  ```
  [[define_function(search_from_api) Searches an API for a query and returns results]]
  ```

**Note**: The description should ideally hint at whether the function is code-based or agent-based to clarify its handling in the system.

## Tips for Success
- Use unique function names to prevent conflicts.
- Keep descriptions clear and concise, specifying the function’s purpose and type where relevant.
- Refer to the few-shot examples to ensure proper formatting.
