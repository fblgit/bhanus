# ./src/pipelines/scraping_pipeline.py
import logging
from typing import List, Dict, Set, Optional, Any

#from src.mcp_client.client import MCPClient
from src.prompts.utils import load_prompt_template, format_prompt
from src.generation.parser import parse_agent_output
#from src.inference.vllm_engine import VLLMEngine
from src.generation.interceptor import GenerationInterceptor

# Configure logging
logger = logging.getLogger(__name__)

class ScrapingPipeline:
    """Manages a web scraping workflow using dynamically selected MCP tools."""

    def __init__(self, mcp_client: Any, engine: Any = None, verbose: bool = False, log_level: str = "INFO"):
        """
        Initialize the ScrapingPipeline with an MCP client and optional VLLM engine.

        Args:
            mcp_client (MCPClient): Client to invoke tools on the MCP server.
            engine (VLLMEngine, optional): Engine for agent-based tool selection if provided.
            verbose (bool): If True, log detailed information about pipeline execution.
            log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING").
        """
        self.mcp_client = mcp_client
        self.engine = engine
        self.verbose = verbose
        self.interceptor = None
        if engine:
            self.interceptor = GenerationInterceptor(mcp_client, engine, verbose=verbose, log_level=log_level)
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                           format="%(asctime)s - %(levelname)s - %(message)s")
        if self.verbose:
            logger.info("Initialized ScrapingPipeline with MCPClient and %s engine", "active" if engine else "no")

    def execute(self, goal: str, selected_tools: List[str] = None, max_urls: int = 3) -> List[Dict]:
        """
        Execute the scraping pipeline for a given goal using dynamically selected tools.

        Args:
            goal (str): The task goal (e.g., "scrape and summarize AI news").
            selected_tools (List[str], optional): Pre-selected tools from tools_picker; if None, dynamically select.
            max_urls (int): Maximum number of URLs to process (default: 3).

        Returns:
            List[Dict]: List of dictionaries with 'url' and 'summary' keys.
        """
        summaries: List[Dict] = []
        if not goal.strip():
            logger.warning("Goal is empty; returning empty summaries")
            return summaries

        try:
            if self.verbose:
                logger.info("Starting pipeline for goal: '%s', max_urls: %d", goal, max_urls)

            # Load all available tools, including registered ones
            all_tools = self.mcp_client.list_tools(include_local=True)
            available_tool_names = {tool["name"] for tool in all_tools}

            # Step 1: Dynamically select tools if not provided
            if selected_tools is None and self.engine and self.interceptor:
                if self.verbose:
                    logger.info("No tools provided; using tools_picker to select dynamically")
                tools_template = load_prompt_template("tools_picker", verbose=self.verbose)
                tools_prompt = format_prompt(tools_template, task=goal, tools=all_tools)
                generator = self.engine.generate_with_interception(tools_prompt, self.interceptor, max_tokens=200)
                tools_result = "".join(list(generator)).strip()
                parsed_tools = parse_agent_output(tools_result, "tools_picker", engine=self.engine, verbose=self.verbose)
                if not parsed_tools or "selected_tools" not in parsed_tools:
                    raise RuntimeError(f"tools_picker failed to return valid tool list: '{tools_result}'")
                selected_tools = parsed_tools["selected_tools"]
                if self.verbose:
                    logger.info("Dynamically selected tools: %s", selected_tools)
            elif selected_tools is None:
                logger.warning("No tools provided and no engine available; using default tools")
                selected_tools = ["search_from_api", "parse_html", "summarise_text"]

            # Validate tool availability
            missing_tools = set(selected_tools) - available_tool_names
            if missing_tools:
                logger.warning("Missing tools: %s; proceeding with available tools", missing_tools)
                selected_tools = [t for t in selected_tools if t in available_tool_names]
                if not selected_tools:
                    raise RuntimeError("No available tools after filtering; pipeline cannot proceed")

            # Step 2: Search for URLs
            search_tool = next((t for t in selected_tools if "search" in t.lower()), None)
            if not search_tool:
                raise RuntimeError("No search tool available in selected tools")
            search_result = self._invoke_tool(search_tool, {"query": goal})
            urls = self._extract_urls(search_result, search_tool)
            if not urls:
                logger.warning("No URLs found for goal")
                return summaries
            urls = urls[:max_urls]
            if self.verbose:
                logger.debug("Found %d URLs: %s", len(urls), urls)

            # Step 3: Process each URL with selected tools
            parse_tool = next((t for t in selected_tools if "parse" in t.lower()), None)
            summarize_tool = next((t for t in selected_tools if "summarise" in t.lower() or "summarize" in t.lower()), None)
            if not parse_tool or not summarize_tool:
                raise RuntimeError(f"Required tools missing: parse_tool={parse_tool}, summarize_tool={summarize_tool}")

            scraping_template = load_prompt_template("scraping", verbose=self.verbose)
            for url in urls:
                try:
                    if self.verbose:
                        logger.info("Processing URL: %s", url)

                    # Scrape content
                    scrape_result = self._invoke_tool(parse_tool, {"url": url})
                    content = self._extract_content(scrape_result, parse_tool, url)
                    if not content:
                        logger.warning("No content retrieved for %s", url)
                        summaries.append({"url": url, "summary": "No content available"})
                        continue

                    # Summarize content
                    summary_prompt = format_prompt(scraping_template, url=url, content=content)
                    if self.engine and self.interceptor:
                        generator = self.engine.generate_with_interception(summary_prompt, self.interceptor, max_tokens=200)
                        summary = "".join(list(generator)).strip()
                    else:
                        summary_result = self._invoke_tool(summarize_tool, {"text": content})
                        summary = self._extract_summary(summary_result, summarize_tool)
                    summaries.append({"url": url, "summary": summary})
                    if self.verbose:
                        logger.debug("Summary for %s: %s", url, summary)

                except RuntimeError as e:
                    logger.error("Error processing %s: %s", url, e)
                    summaries.append({"url": url, "summary": f"Error: {str(e)}"})

            if self.verbose:
                logger.info("Pipeline completed with %d summaries", len(summaries))
            return summaries

        except Exception as e:
            error_msg = f"Pipeline execution failed for goal '{goal}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _invoke_tool(self, tool_name: str, params: Dict) -> any:
        """
        Invoke a tool on the MCP server with flexible return type handling.

        Args:
            tool_name (str): Name of the tool to invoke.
            params (Dict): Parameters required by the tool.

        Returns:
            any: Raw result from the tool invocation, adapted to a usable format.
        """
        try:
            if self.verbose:
                logger.debug("Invoking tool '%s' with params: %s", tool_name, params)
            result = self.mcp_client.invoke_tool(tool_name, params, consent=True)
            if self.verbose:
                logger.debug("Tool '%s' raw result: %s", tool_name, result)
            # Adapt result to a usable format
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                return {"result": result}
            elif isinstance(result, list):
                return {"items": result}
            else:
                logger.warning("Unexpected result type from '%s': %s; wrapping as string", tool_name, type(result))
                return {"result": str(result)}
        except Exception as e:
            error_msg = f"Failed to invoke tool '{tool_name}' with params {params}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _extract_urls(self, result: any, tool_name: str) -> List[str]:
        """
        Extract URLs from a search tool result, handling various formats.

        Args:
            result (any): Raw result from the search tool.
            tool_name (str): Name of the tool for logging context.

        Returns:
            List[str]: List of URLs extracted from the result.
        """
        if isinstance(result, dict):
            if 'results' in result:
                result = result.get('results')
            urls = result.get("urls", result.get("result", result.get("items", [])))
            if isinstance(urls, str):
                urls = [urls]
            elif not isinstance(urls, list):
                logger.warning("Invalid URL format from '%s': %s", tool_name, urls)
                return []
            return [url for url in urls if isinstance(url, str)]
        elif isinstance(result, list):
            return [item for item in result if isinstance(item, str)]
        elif isinstance(result, str):
            return [result]
        logger.warning("No URLs extractable from '%s' result: %s", tool_name, result)
        return []

    def _extract_content(self, result: any, tool_name: str, url: str) -> Optional[str]:
        """
        Extract content from a parse tool result, handling various formats.

        Args:
            result (any): Raw result from the parse tool.
            tool_name (str): Name of the tool for logging context.
            url (str): URL being processed for error reporting.

        Returns:
            Optional[str]: Extracted content or None if invalid.
        """
        if isinstance(result, dict):
            content = result.get("content", result.get("result", result.get("text")))
            if isinstance(content, str):
                return content
            logger.warning("Invalid content format from '%s' for %s: %s", tool_name, url, content)
            return None
        elif isinstance(result, str):
            return result
        logger.warning("No content extractable from '%s' result for %s: %s", tool_name, url, result)
        return None

    def _extract_summary(self, result: any, tool_name: str) -> str:
        """
        Extract summary from a summarize tool result, handling various formats.

        Args:
            result (any): Raw result from the summarize tool.
            tool_name (str): Name of the tool for logging context.

        Returns:
            str: Extracted summary or error message if invalid.
        """
        if isinstance(result, dict):
            summary = result.get("summary", result.get("result", result.get("text")))
            if isinstance(summary, str):
                return summary
            logger.warning("Invalid summary format from '%s': %s", tool_name, summary)
            return "No summary available"
        elif isinstance(result, str):
            return result
        logger.warning("No summary extractable from '%s' result: %s", tool_name, result)
        return "No summary available"

# Example usage for testing
if __name__ == "__main__":
    class MockMCPClient:
        def list_tools(self, include_local: bool = False) -> List[Dict]:
            return [
                {"name": "search_from_api", "description": "Search", "input": {"query": "string"}, "output": "dict", "type": "code"},
                {"name": "parse_html", "description": "Parse", "input": {"url": "string"}, "output": "dict", "type": "code"},
                {"name": "summarise_text", "description": "Summarize", "input": {"text": "string"}, "output": "dict", "type": "code"},
                {"name": "tools_picker", "description": "Pick tools", "input": {"task": "string"}, "output": "string", "type": "agent"}
            ]

        def invoke_tool(self, tool_name: str, params: Dict, consent: bool) -> any:
            if not consent:
                raise RuntimeError("Consent required")
            if tool_name == "search_from_api":
                return ["http://example.com", "http://test.com"]  # List instead of dict
            elif tool_name == "parse_html":
                return f"Content from {params['url']}"  # String instead of dict
            elif tool_name == "summarise_text":
                return {"summary": f"Summary of {params['text']}"}
            elif tool_name == "tools_picker":
                return {"selected_tools": ["search_from_api", "parse_html", "summarise_text"]}
            return "mock response"  # Unexpected type

    class MockVLLMEngine:
        def generate_with_interception(self, prompt: str, interceptor, **kwargs) -> List[str]:
            return [prompt]  # Simplified for mock

    # Test the pipeline
    client = MockMCPClient()
    engine = MockVLLMEngine()
    pipeline = ScrapingPipeline(client, engine, verbose=True, log_level="DEBUG")
    result = pipeline.execute("scrape and summarize AI news", max_urls=2)
    for summary in result:
        print(f"URL: {summary['url']}, Summary: {summary['summary']}")
