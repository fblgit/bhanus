# tests/test_generation.py

import unittest
from unittest.mock import MagicMock
from src.generation.interceptor import GenerationInterceptor
from src.generation.parser import extract_function_call
from src.mcp_client.client import MCPClient

class TestGenerationInterceptor(unittest.TestCase):
    def setUp(self):
        self.mcp_client = MagicMock(spec=MCPClient)
        self.interceptor = GenerationInterceptor(self.mcp_client)

    def test_extract_function_call(self):
        text = 'Some text <function_call>{"name": "get_time", "params": {}}</function_call> more text'
        func_call = extract_function_call(text)
        self.assertEqual(func_call, {"name": "get_time", "params": {}})

    def test_intercept_stream(self):
        # Mock generator yielding tokens
        def mock_generator():
            yield 'Hello, <function_call>{"name": "get_time", "params": {}}</function_call> world!'

        self.mcp_client.invoke_tool.return_value = {"time": "12:00"}
        intercepted = list(self.interceptor.intercept_stream(mock_generator()))

        expected = [
            'Hello, <function_call>{"name": "get_time", "params": {}}</function_call>',
            '<function_result>{"status": "success", "result": {"time": "12:00"}}</function_result>',
            ' world!'
        ]
        self.assertEqual(intercepted, expected)

if __name__ == "__main__":
    unittest.main()
