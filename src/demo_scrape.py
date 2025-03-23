# ./src/main.py
import time
import logging
import yaml
import threading
from typing import List, Dict

from src.inference.vllm_engine import VLLMEngine
from src.mcp_client.client import MCPClient
from src.mcp_server.mcp_server import MCPServer  # New import for MCP server
from src.generation.interceptor import GenerationInterceptor
from src.pipelines.scraping_pipeline import ScrapingPipeline
from src.prompts.utils import load_prompt_template, format_prompt
from src.prompts.validator import PromptValidator
from src.generation.parser import parse_agent_output

# Configure logging
logger = logging.getLogger(__name__)

def main():
    """Orchestrate the MCP-vLLM system with agent-driven workflows and function registration."""
    try:
        # Load configuration
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict) or "vllm" not in config or "mcp" not in config:
            raise ValueError("Invalid config.yaml: must contain 'vllm' and 'mcp' sections")

        model_name = config["vllm"]["model"]
        mcp_host = config["mcp"].get("host", "0.0.0.0")
        mcp_port = config["mcp"].get("port", 6000)
        mcp_server_url = f"http://{mcp_host}:{mcp_port}"
        log_level = config.get("logging", {}).get("level", "DEBUG")
        registry_file = config["registry"]["path"]

        # Initialize and start the MCP server
        logger.info("Starting MCP server...")
        server = MCPServer(
            host=mcp_host,
            port=mcp_port,
            #registry_file=registry_file,
            verbose=True,
            log_level=log_level
        )
        # Run server in a background thread to allow main execution to continue
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        logger.info("MCP server running on %s:%d", mcp_host, mcp_port)

        # Initialize core components
        logger.info("Initializing core components...")
        engine = VLLMEngine(config_path="config/config.yaml", verbose=True)
        engine.load_model(model_name)
        time.sleep(5)

        server.engine = engine
        mcp_client = MCPClient(config_path="config/config.yaml", engine=engine, verbose=True)
        mcp_client.connect(mcp_server_url)

        interceptor = GenerationInterceptor(mcp_client, engine, verbose=True, log_level=log_level)
        validator = PromptValidator(engine=engine, verbose=True)  # Pass engine for registry access

        # Phase 1: Define and register a new function
        logger.info("Defining and registering a new function...")
        define_template = load_prompt_template("define_function", verbose=True)
        define_prompt = format_prompt(
            define_template,
            name="web_search_summary",
            description="Searches in Google, Fetches the Text of each result and create a summary of each link.",
            input={"query": "string"},
            output="string",
            type="agent"
        )
        generator = engine.generate_with_interception(define_prompt, interceptor, max_tokens=100)
        define_result = "".join(list(generator)).strip()
        expected_format = r"\[registered\]web_search_summary\[/registered\]"
        if not validator.validate(define_result, expected_format, template_name="define_function"):
            raise ValueError(f"Function definition failed: expected '[registered]web_search_summary[/registered]', got '{define_result}'")
        logger.info("Function registration result: '%s'", define_result)

        # Phase 2: Generate agent list for the task using agent_factory
        logger.info("Generating agent list with agent_factory...")
        goals = ["fetch and iterate over google search results urls", "write a report with the results summaries"]
        for goal in goals:
            agent_template = load_prompt_template("agent_factory", verbose=True)
            agent_prompt = format_prompt(agent_template, task=goal, agents=mcp_client.list_tools(include_local=True))
            generator = engine.generate_with_interception(agent_prompt, interceptor, max_tokens=200)
            agent_result = "".join(list(generator)).strip()
            parsed_agents = { 'agents': f'{agent_result}'.split('\n')[0].split(']')[0].replace(',','').replace(']', '').replace('[', '').split(' ') }
            #parsed_agents = parse_agent_output(agent_result, "agent_factory", engine=engine, verbose=True)
            if not parsed_agents or "agents" not in parsed_agents:
                raise ValueError(f"Agent factory failed: '{agent_result}'")
            logger.info("Agent list: %s", parsed_agents["agents"])

        # Phase 3: Select tools dynamically with tools_picker
        for goal in goals:
            logger.info("Selecting tools with tools_picker...")
            tools_template = load_prompt_template("tools_picker", verbose=True)
            tools_prompt = format_prompt(tools_template, task=goal, tools=mcp_client.list_tools(include_local=True))
            generator = engine.generate_with_interception(tools_prompt, interceptor, max_tokens=200)
            tools_result = "".join(list(generator)).strip()
            parsed_tools = { 'selected_tools': f'{tools_result}'.split('\n')[0].split(']')[0].replace(',','').replace(']', '').replace('[', '').split(' ') }
            #parsed_tools = parse_agent_output(tools_result, "tools_picker", engine=engine, verbose=True)
            if not parsed_tools or "selected_tools" not in parsed_tools:
                raise ValueError(f"Tools picker failed: '{tools_result}'")
            selected_tools = parsed_tools["selected_tools"]
            logger.info("Selected tools: %s", selected_tools)

        # Phase 4: Execute scraping pipeline with dynamic tools
        logger.info("Executing scraping pipeline with dynamic tools...")
        goal = "Search on google the latest AI news and write a report"
        pipeline = ScrapingPipeline(mcp_client, engine, verbose=True)
        selected_tools = ['google_search', 'fetch_text_url', 'report_writer']
        #summaries = pipeline.execute(goal, selected_tools=None, max_urls=2)
        summaries = pipeline.execute(goal, selected_tools=selected_tools, max_urls=2)
        if not summaries:
            logger.warning("No summaries generated from the pipeline")
            return
        else:
            logger.info("Scraping results:")
            for summary in summaries:
                print(f"URL: {summary['url']}\nSummary: {summary['summary']}\n")

        # Phase 5: Use registered agent to process summaries
        logger.info("Processing summaries with registered report_writer...")
        tools = mcp_client.list_tools(include_local=True)
        if any(t["name"] == "report_writer" for t in tools):
            report_template = load_prompt_template("report_writer", verbose=True)
            report_prompt = format_prompt(report_template, summaries=summaries)
            generator = engine.generate_with_interception(report_prompt, interceptor, max_tokens=500)
            report_result = "".join(list(generator)).strip()
            expected_format = report_template.get("expected_format", ".*")
            if not validator.validate(report_result, expected_format, template_name="report_writer"):
                logger.warning("Report writer output invalid: '%s'", report_result)
            else:
                logger.info("Generated Report: %s", report_result)
                print(f"Generated Report: {report_result}")
        else:
            logger.warning("report_writer not found; skipping report generation")

    except Exception as e:
        logger.error("Execution failed: %s", str(e), exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    # Set up logging for main execution
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
