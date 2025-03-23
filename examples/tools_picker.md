# Tools Picker Template Specification

## Overview
The `tools_picker.jinja2` template enables an AI model to select appropriate tools from a provided list based on a task description. It produces a structured string output that can be parsed and validated, supporting dynamic tool selection in workflows.

## Purpose
- Facilitate agent-driven tool selection for tasks like data processing or web scraping.
- Ensure compatibility with systems that rely on structured outputs (e.g., `interceptor.py`, `scraping_pipeline.py`).

## Inputs
- **task** (str): The task description (e.g., "fetch and analyze stock prices").
- **tools** (list[dict]): A list of tools, each with:
  - `name` (str): Tool name (e.g., "fetch_stock_data").
  - `description` (str): What the tool does.
  - `input` (dict): Expected input parameters.
  - `output` (str): Output type (e.g., "JSON").
  - `type` (str): Tool category (e.g., "code", "agent").

## Expected Output
- Format:
  ```
  Task: <task description>
  Selected tools: [<tool1>, <tool2>, ...]
  ```
  - `<task description>`: The input task, verbatim.
  - `[<tool1>, <tool2>, ...]`: A list of selected tool names in square brackets, comma-separated.

## Validation
- Must match the regex: `Task: .+\\nSelected tools: \\[.*\\]`.
- Ensures the output is parseable by downstream systems.

## Few-Shot Examples

### Example 1
- **Task**: "Get the current time and weather for New York."
- **Available Tools**:
  - `get_current_time`: Gets the current time.
    - Input: {}
    - Output: "string"
    - Type: "code"
  - `get_weather`: Gets the weather for a location.
    - Input: {"location": "str"}
    - Output: "JSON"
    - Type: "code"
- **Output**:
  ```
  Task: Get the current time and weather for New York.
  Selected tools: [get_current_time, get_weather]
  ```

### Example 2
- **Task**: "Summarize a webpage and save the summary to a file."
- **Available Tools**:
  - `parse_html`: Parses HTML content.
    - Input: {"url": "str"}
    - Output: "string"
    - Type: "code"
  - `summarise_text`: Summarizes text.
    - Input: {"text": "str"}
    - Output: "string"
    - Type: "agent"
  - `write_to_file`: Writes content to a file.
    - Input: {"content": "str", "filename": "str"}
    - Output: "None"
    - Type: "code"
- **Output**:
  ```
  Task: Summarize a webpage and save the summary to a file.
  Selected tools: [parse_html, summarise_text, write_to_file]
  ```

### Example 3
- **Task**: "Search for AI news and generate a report."
- **Available Tools**:
  - `search_from_api`: Searches for content via API.
    - Input: {"query": "str"}
    - Output: "JSON"
    - Type: "code"
  - `generate_report`: Generates a report from data.
    - Input: {"data": "JSON"}
    - Output: "string"
    - Type: "agent"
- **Output**:
  ```
  Task: Search for AI news and generate a report.
  Selected tools: [search_from_api, generate_report]
  ```

## Notes
- The model should select only tools directly relevant to the task.
- If no tools apply, return `Selected tools: []`.
- Examples are critical since the format isn’t native to the model—consistency relies on these samples.
