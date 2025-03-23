# tests/test_pipelines.py

import unittest
from unittest.mock import MagicMock
from src.pipelines.scraping_pipeline import ScrapingPipeline
from src.mcp_client.client import MCPClient

class TestScrapingPipeline(unittest.TestCase):
    def setUp(self):
        self.mcp_client = MagicMock(spec=MCPClient)
        self.pipeline = ScrapingPipeline(self.mcp_client)

    def test_execute_success(self):
        # Mock search tool
        self.mcp_client.invoke_tool.side_effect = [
            {"urls": ["http://example.com"]},  # search_from_api
            {"content": "Sample content"},     # parse_html
            {"summary": "Sample summary"}      # summarise_text
        ]

        summaries = self.pipeline.execute("test query", max_urls=1)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["url"], "http://example.com")
        self.assertEqual(summaries[0]["summary"], "Sample summary")

    def test_execute_tool_failure(self):
        # Mock search tool to return URLs, but scraping fails
        self.mcp_client.invoke_tool.side_effect = [
            {"urls": ["http://example.com"]},  # search_from_api
            Exception("Scraping failed")       # parse_html
        ]

        summaries = self.pipeline.execute("test query", max_urls=1)
        self.assertEqual(len(summaries), 0)  # No summaries due to error

if __name__ == "__main__":
    unittest.main()
