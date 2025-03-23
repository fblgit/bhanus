# MCP-vLLM Integration Project
A production-grade system integrating MCP with vLLM for efficient LLM inference and tool execution.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure: Edit `config/config.yaml`


## Current Progress:

- [X] Inference Engine vLLM
- [X] Template Engine
- [X] MCP Client
- [X] MCP Server
- [X] Primitive Functions & Handlers
- [X] Agent & Function Registration
- [X] Prompt Templates
- [ ] Parse Pipeline (WIP)
- [ ] Ensure Agents & Functions Existence or Trigger Creation
- [ ] E2E Flow (WIP `main.py`)
- [ ] 

2025-03-23 11:50:26,023 - INFO - Starting MCP server...
2025-03-23 11:50:26,026 - INFO - Initialized MCPServer on 127.0.0.1:6000 with registry function_registry.json
2025-03-23 11:50:26,026 - INFO - Starting MCP server on 127.0.0.1:6000
2025-03-23 11:50:26,026 - INFO - MCP server running on 127.0.0.1:6000
2025-03-23 11:50:26,027 - INFO - Initializing core components...
 * Serving Flask app 'src.mcp_server.mcp_server'
 * Debug mode: off
2025-03-23 11:50:26,033 - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:6000
2025-03-23 11:50:26,033 - INFO - Press CTRL+C to quit
INFO 03-23 11:50:28 llm_engine.py:226] Initializing an LLM engine (v0.6.1.dev238+ge2c6e0a82) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=Qwen/Qwen2.5-1.5B-Instruct, use_v2_block_manager=False, num_scheduler_steps=1, multi_step_stream_outputs=False, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None)
INFO 03-23 11:50:30 model_runner.py:1014] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
INFO 03-23 11:50:31 weight_utils.py:242] Using model weights format ['*.safetensors']
INFO 03-23 11:50:31 weight_utils.py:287] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.14it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.14it/s]

INFO 03-23 11:50:32 model_runner.py:1025] Loading model weights took 2.8875 GB
INFO 03-23 11:50:32 gpu_executor.py:122] # GPU blocks: 281784, # CPU blocks: 9362
INFO 03-23 11:50:35 model_runner.py:1329] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 03-23 11:50:35 model_runner.py:1333] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 03-23 11:50:40 model_runner.py:1456] Graph capturing finished in 5 secs.
2025-03-23 11:50:41,053 - INFO - Loaded model: Qwen/Qwen2.5-1.5B-Instruct with kwargs: {}
2025-03-23 11:50:46,057 - INFO - 127.0.0.1 - - [23/Mar/2025 11:50:46] "GET / HTTP/1.1" 405 -
2025-03-23 11:50:46,058 - INFO - Connected to MCP server at http://127.0.0.1:6000
2025-03-23 11:50:46,058 - INFO - Defining and registering a new function...
2025-03-23 11:50:46,060 - INFO - Loaded template 'define_function.jinja2' with expected_format '\[\[define_function\(.*\).*?\]\]' and description 'Define a new MCP function for registration'
2025-03-23 11:50:46,060 - INFO - Generating with interception for prompt: '

### System:
To define a new function for the MCP system, use the following syntax:

Example 1: Code-Based Function
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

Example 2: Agent-Based Function
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

Example 3: Function with Parameters
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
### Instruction:
Define a new function for the MCP system:

Where:
- "report_writer" is the name of the function (e.g., "report_writer").
- "Writes a report from summarized content as an agent" is a brief description of what the function does.

### Response:
'

Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 13.89it/s, est. speed input: 5004.87 toks/s, output: 250.19 toks/s]
2025-03-23 11:50:46,138 - INFO - Detected function definition: report_writer with {'description': 'Writes a report from summarized content as an agent', 'input': {}, 'output': 'string', 'type': 'agent'}
2025-03-23 11:50:46,138 - INFO - Registering tool 'report_writer' with definition: {'name': 'report_writer', 'description': 'Writes a report from summarized content as an agent', 'input': {}, 'output': 'string', 'type': 'agent'}
2025-03-23 11:50:46,141 - INFO - Registered tool: {'name': 'report_writer', 'description': 'Writes a report from summarized content as an agent', 'input': {}, 'output': 'string', 'type': 'agent'}
2025-03-23 11:50:46,141 - INFO - 127.0.0.1 - - [23/Mar/2025 11:50:46] "POST / HTTP/1.1" 200 -
2025-03-23 11:50:46,142 - INFO - Registered/updated function locally: report_writer with definition: {'name': 'report_writer', 'description': 'Writes a report from summarized content as an agent', 'input': {}, 'output': 'string', 'type': 'agent'}
2025-03-23 11:50:46,142 - INFO - Successfully registered tool 'report_writer' on server and locally
2025-03-23 11:50:46,142 - INFO - Registered/updated function locally: report_writer with definition: {'name': 'report_writer', 'description': 'Writes a report from summarized content as an agent', 'input': {}, 'output': 'string', 'type': 'agent'}
2025-03-23 11:50:46,142 - INFO - Function registration result: '```
[registered]report_writer[/registered]'
2025-03-23 11:50:46,143 - INFO - Generating agent list with agent_factory...
2025-03-23 11:50:46,143 - INFO - Loaded template 'agent_factory.jinja2' with expected_format 'Goal: .+\\nAgents: \\[.*\\]' and description 'Generate a list of collaborating agents for a given goal'
2025-03-23 11:50:46,143 - INFO - Generating with interception for prompt: '

You are an AI assistant tasked with generating a list of expert agents that can collaborate to achieve a specific goal. The goal is provided, and you must output a list of agents that are best suited to work together on this task.

**Instructions**:
- Always start with the line: "Goal: <task>"
- Then, list the agents in the format: "Agents: [agent1, agent2, ...]"
- Ensure the agents are relevant to the task and can collaborate effectively.

**Examples**:

1. **Input**: task = "summarize code and write a report"
   **Output**:
   Goal: summarize code and write a report
   Agents: [code_analyzer, summarizer, report_writer]

2. **Input**: task = "plan a trip to Paris"
   **Output**:
   Goal: plan a trip to Paris
   Agents: [travel_planner, accommodation_finder, activity_suggestor]

3. **Input**: task = "debug a software application"
   **Output**:
   Goal: debug a software application
   Agents: [debugger, tester, log_analyzer]

**Your Task**:
Generate the list of agents for the following goal:

Goal: summarize code and write a report
Existing Agents:
Agents: ['
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.53it/s, est. speed input: 403.58 toks/s, output: 305.74 toks/s]
2025-03-23 11:50:46,800 - INFO - Agent list: ['code_analyzer', 'summarizer', 'report_writer']
2025-03-23 11:50:46,800 - INFO - Selecting tools with tools_picker...
2025-03-23 11:50:46,801 - INFO - Loaded template 'tools_picker.jinja2' with expected_format 'Task: .+\\nSelected tools: \\[.*\\]' and description 'Select tools for a task'
2025-03-23 11:50:46,802 - INFO - Requesting list of tools from MCP server at http://127.0.0.1:6000
2025-03-23 11:50:46,803 - INFO - 127.0.0.1 - - [23/Mar/2025 11:50:46] "POST / HTTP/1.1" 200 -
2025-03-23 11:50:46,804 - INFO - Generating with interception for prompt: '

### System:
You are an AI assistant tasked with selecting the most relevant tools for a given task from a list of available tools. Your goal is to analyze the task and choose the tools that are necessary to accomplish it effectively.

**Available Tools:**

- **fetch_url**
  - Description: Fetch content from a URL via GET request
  - Input: {"url": "string"}
  - Output: string
  - Type: code

- **fetch_text_url**
  - Description: Fetch URL content and extract readable text using BeautifulSoup
  - Input: {"url": "string"}
  - Output: string
  - Type: code

- **fetch_links_url**
  - Description: Fetch URL content and extract links with their text using BeautifulSoup
  - Input: {"url": "string"}
  - Output: dict
  - Type: code

- **pdf2text**
  - Description: Convert a PDF file to text given its file path
  - Input: {"file_path": "string"}
  - Output: string
- **fetch_file**
  - Description: Fetch a file from a URL and store it temporarily, returning the file path
  - Input: {"url": "string"}
  - Output: string
  - Type: code

- **read_file**
  - Description: Read contents of a file with optional chunking
  - Input: {"end": "int", "file_path": "string", "limit": "int", "start": "int"}
  - Output: string
  - Type: code

- **write_file**
  - Description: Write content to a file and return its path, with optional file path override
  - Input: {"file_content": "string", "file_path": "string"}
  - Output: string
  - Type: code

- **run_snippet**
  - Description: Run Python code from content in a separate thread and return file path and output
  - Input: {"code_content": "string"}
  - Output: dict
  - Type: code

- **google_search**
  - Description: Perform a Google search and return results
  - Input: {"limit": "int", "query": "string", "start": "int"}
  - Output: dict
  - Type: code

- **calculator**
  - Description: Evaluate a mathematical expression and return the result
  - Input: {"expression": "string"}
  - Output: float
  - Type: code

- **report_writer**
  - Description: Writes a report from summarized content as an agent
  - Input: {}
  - Output: string
  - Type: agent


### Instruction:
1. Review the task description provided.
2. Examine the list of available tools, each with their name, description, input, output, and type.
3. Select the tools that are most appropriate for the task. You may select multiple tools if needed.
4. Output your selection in the following format:
   ```
   Task: <task description>
   Selected tools: [<tool1>, <tool2>, ...]
   ```
   - Replace `<task description>` with the provided task.
   - Replace `[<tool1>, <tool2>, ...]` with the names of the selected tools, enclosed in square brackets and separated by commas.
Based on the task and available tools provided, generate the output in the specified format.
**Task Description:**
summarize code and write a report

### Response:
Task: summarize code and write a report
Selected tools: ['

Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.16it/s, est. speed input: 20126.74 toks/s, output: 217.51 toks/s]
2025-03-23 11:50:46,844 - INFO - Selected tools: ['run_snippet', 'report_writer', 'google_search']
2025-03-23 11:50:46,844 - INFO - Executing scraping pipeline with dynamic tools...
2025-03-23 11:50:46,844 - INFO - Initialized ScrapingPipeline with MCPClient and active engine
2025-03-23 11:50:46,844 - INFO - Starting pipeline for goal: 'summarize code and write a report', max_urls: 2
2025-03-23 11:50:46,844 - INFO - Requesting list of tools from MCP server at http://127.0.0.1:6000
2025-03-23 11:50:46,845 - INFO - 127.0.0.1 - - [23/Mar/2025 11:50:46] "POST / HTTP/1.1" 200 -
2025-03-23 11:50:46,846 - INFO - Invoking tool 'google_search' with params: {'query': 'summarize code and write a report'} and consent: True
2025-03-23 11:50:46,848 - INFO - Invoking tool 'google_search' with params: {'consent': True, 'query': 'summarize code and write a report'}
2025-03-23 11:50:47,563 - INFO - 127.0.0.1 - - [23/Mar/2025 11:50:47] "POST / HTTP/1.1" 200 -
2025-03-23 11:50:47,564 - WARNING - Invalid URL format from 'google_search': {'results': [{'title': 'Highly Efficient Prompt for Summarizing — GPT-4 : r/ChatGPTPro', 'url': 'https://www.reddit.com/r/ChatGPTPro/comments/13n55w7/highly_efficient_prompt_for_summarizing_gpt4/'}, {'title': 'Overview of Copilot for Power BI - Power BI | Microsoft Learn', 'url': 'https://learn.microsoft.com/en-us/power-bi/create-reports/copilot-introduction'}, {'title': 'How to write a good Test Summary Report? | BrowserStack', 'url': 'https://www.browserstack.com/guide/how-to-write-test-summary-report'}, {'title': 'Create a grouped or summary report - Microsoft Support', 'url': 'https://support.microsoft.com/en-us/office/create-a-grouped-or-summary-report-f23301a1-3e0a-4243-9002-4a23ac0fdbf3'}, {'title': 'O*NET OnLine Help: Summary Report', 'url': 'https://www.onetonline.org/help/online/summary'}]}
2025-03-23 11:50:47,564 - WARNING - No URLs found for goal
2025-03-23 11:50:47,564 - WARNING - No summaries generated from the pipeline
2025-03-23 11:50:47,564 - INFO - Processing summaries with registered report_writer...
2025-03-23 11:50:47,564 - INFO - Requesting list of tools from MCP server at http://127.0.0.1:6000
2025-03-23 11:50:47,566 - INFO - 127.0.0.1 - - [23/Mar/2025 11:50:47] "POST / HTTP/1.1" 200 -
2025-03-23 11:50:47,567 - INFO - Loaded template 'report_writer.jinja2' with expected_format 'Report:.*' and description 'Generate a report from summaries'
2025-03-23 11:50:47,567 - WARNING - Unsupported or unmatched expected_format: 'Report:.*' for output: '
```
