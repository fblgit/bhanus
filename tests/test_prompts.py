# tests/test_prompts.py

import unittest
from src.prompts.validator import PromptValidator

class TestPromptValidator(unittest.TestCase):
    def test_toolbox_catalog_valid(self):
        output = 'Here is the toolbox catalog: {"tools": [{"name": "get_time", "description": "Gets time", "input": {}, "output": "string"}]}'
        expected_format = "Here is the toolbox catalog: {\"tools\": [{\"name\": \"...\", \"description\": \"...\", \"input\": {...}, \"output\": \"...\"}]}."
        self.assertTrue(PromptValidator.validate(output, expected_format))

    def test_toolbox_catalog_invalid(self):
        output = 'Invalid output'
        expected_format = "Here is the toolbox catalog: {\"tools\": [{\"name\": \"...\", \"description\": \"...\", \"input\": {...}, \"output\": \"...\"}]}."
        self.assertFalse(PromptValidator.validate(output, expected_format))

    def test_toolbox_selector_valid(self):
        output = "Available: [get_time], Missing: []"
        expected_format = "Available: [tool1, tool2], Missing: [tool3]"
        self.assertTrue(PromptValidator.validate(output, expected_format))

    def test_toolbox_selector_invalid(self):
        output = "Available tools: get_time"
        expected_format = "Available: [tool1, tool2], Missing: [tool3]"
        self.assertFalse(PromptValidator.validate(output, expected_format))

    def test_scraping_valid(self):
        output = "Summary of https://example.com: This is a summary."
        expected_format = "Summary of [URL]: [summary text]"
        self.assertTrue(PromptValidator.validate(output, expected_format))

    def test_scraping_invalid(self):
        output = "Scraped content: This is content."
        expected_format = "Summary of [URL]: [summary text]"
        self.assertFalse(PromptValidator.validate(output, expected_format))

if __name__ == "__main__":
    unittest.main()
