## `report_writer.md` Documentation

### Purpose
The `report_writer.md` file is a Markdown-style technical specification that details the template’s purpose, usage, and syntax, with few-shot examples to guide the agent system and developers.

### Content of `report_writer.md`

```markdown
# Report Writer Template

## Purpose
The `report_writer.jinja2` template is designed for the "MCP-vLLM Integration" project. It prompts the model to generate a structured report from a list of summaries, each associated with a URL. The output is formatted as a Markdown list for readability and easy rendering.

## Usage
This template is rendered in the `main.py` script as part of the workflow to process summaries from the scraping pipeline. It uses Jinja2 with the `summaries` variable, a list of dictionaries containing `url` and `summary` keys.

## Inputs
- `summaries` (list[dict]): A list of dictionaries, each with:
  - `url` (str): The URL of the scraped content.
  - `summary` (str): The summary of the content.

## Expected Output
The output is a Markdown-formatted string starting with "Report: Summarized Content" followed by a list of URLs and summaries.

### Output Format
```
Report: Summarized Content

- **URL**: <url>  
  **Summary**: <summary>

- **URL**: <url>  
  **Summary**: <summary>
```

## Few-shot Examples

### Example 1
**Input:**
```python
summaries = [
    {"url": "http://example.com", "summary": "This is a summary of example.com."},
    {"url": "http://test.com", "summary": "This is a summary of test.com."}
]
```

**Expected Output:**
```
Report: Summarized Content

- **URL**: http://example.com  
  **Summary**: This is a summary of example.com.

- **URL**: http://test.com  
  **Summary**: This is a summary of test.com.
```

### Example 2
**Input:**
```python
summaries = [
    {"url": "http://news.com", "summary": "Latest AI news summary."}
]
```

**Expected Output:**
```
Report: Summarized Content

- **URL**: http://news.com  
  **Summary**: Latest AI news summary.
```

## Notes
- The template uses Jinja2’s `for` loop to iterate over the `summaries` list and format each entry.
- The output is validated with a regex pattern (`Report:.*`) to ensure it begins with "Report:".
- Few-shot examples are included in the template as comments and here in the documentation to reinforce the expected syntax.
```

#### Explanation
- **Purpose and Usage**: Explains the template’s role and integration into the project.
- **Inputs and Outputs**: Defines the data structure and format, with examples for clarity.
- **Few-shot Examples**: Provides concrete input-output pairs to guide the model and developers.
- **Notes**: Highlights technical details like Jinja2 usage and validation.
