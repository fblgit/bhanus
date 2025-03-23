# ./setup.py
from setuptools import setup, find_packages

setup(
    name="mcp_vllm_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vllm==0.6.2",          # Stable version as of December 2024; adjust based on vLLM releases
        "pyyaml==6.0.1",        # Retained for config compatibility, though Jinja2 will replace YAML prompts
        "jsonrpcclient==4.0.3",  # Pinned for MCP communication stability
        "jinja2==3.1.4",        # Pinned latest stable as of March 2025 for advanced templating
        "tenacity==8.5.0",      # Pinned for retry logic in MCPClient (already used)
        "requests==2.32.3",     # Pinned for HTTP requests in MCPClient
        "flask==3.0.3",         # Add Flask for MCPServer
        "google>=3.0.0",
        "PyPDF2>=3.0.1",
        "beautifulsoup4>=4.13.3",
        "portalocker>=3.1.1",
        "pytest",
        "pytest-mock"
    ],
    author="JUANAKO.AI",
    description="MCP-vLLM Integration Project: Enables LLMs to dynamically create and register MCP functions, including code and agent-based tools, with persistent storage and Jinja2 templating.",
    long_description="This project integrates the Model Context Protocol (MCP) with vLLM for efficient LLM inference, supporting dynamic function creation and registration (code and agent-based) with permanent storage in a JSON registry. Prompts use Jinja2 templating for flexibility.",
    python_requires=">=3.8",  # Ensure compatibility with modern Python
)
