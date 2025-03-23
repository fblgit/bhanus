# agent_factory.jinja2 Technical Specification

## Purpose
The `agent_factory.jinja2` template instructs an AI model to generate a list of expert agents that can collaborate to achieve a specific goal. This is part of a system where agent-based workflows are dynamically created, and the output must adhere to a strict format for downstream parsing and validation.

## Inputs
- **task** (str): A description of the goal or task (e.g., "summarize code and write a report").

## Expected Output
The output is a two-line string:
- **First line**: "Goal: <task>" (repeats the input task).
- **Second line**: "Agents: [agent1, agent2, ...]" (a list of agent names in square brackets).

### Output Format
```
Goal: <task>
Agents: [agent1, agent2, ...]
```

## Few-Shot Examples

### Example 1
**Input**:
```python
task = "summarize code and write a report"
```
**Expected Output**:
```
Goal: summarize code and write a report
Agents: [code_analyzer, summarizer, report_writer]
```

### Example 2
**Input**:
```python
task = "plan a trip to Paris"
```
**Expected Output**:
```
Goal: plan a trip to Paris
Agents: [travel_planner, accommodation_finder, activity_suggestor]
```

### Example 3
**Input**:
```python
task = "debug a software application"
```
**Expected Output**:
```
Goal: debug a software application
Agents: [debugger, tester, log_analyzer]
```

## Usage
- The template is rendered with a `task` value to create a prompt for the model.
- The model’s output is parsed (e.g., by `parser.py`) into a structured format like `{"agents": ["agent1", "agent2", ...]}`.
- The output is validated against the regex `Goal: .+\\nAgents: \\[.*\\]` to ensure correctness.

## Notes
- Since the syntax isn’t natively reinforced in the models, the few-shot examples in the template are essential for guiding the model to produce the correct format.
- Agent names should be concise, descriptive, and relevant to the task.
```

### Explanation
- **Purpose and Inputs**: Defines what the template does and what it expects as input.
- **Expected Output**: Specifies the exact two-line format, making it easy to understand and replicate.
- **Few-Shot Examples**: Provides three concrete input-output pairs to reinforce the syntax, mirroring the examples in the template.
- **Usage and Notes**: Explains how the template fits into the system and emphasizes the reliance on examples due to the non-native syntax.

