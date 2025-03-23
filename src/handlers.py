# src/handlers.py
import requests
import os
import tempfile
import subprocess
import threading
import logging
from typing import Dict, Any, Callable
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from googlesearch import search

logger = logging.getLogger(__name__)

# Global registry for handler metadata
HANDLER_REGISTRY = {}

def register_handler(name: str, description: str, input_schema: Dict[str, str], output: str) -> Callable:
    """
    Decorator to register a handler with metadata for automatic inclusion in the MCP server.

    Args:
        name (str): Unique name of the handler/tool.
        description (str): Brief description of the handler's purpose.
        input_schema (Dict[str, str]): Dictionary of parameter names and their types (e.g., {"url": "string"}).
        output (str): Type of the output (e.g., "string", "dict").
    """
    def decorator(func: Callable) -> Callable:
        HANDLER_REGISTRY[name] = {
            "name": name,
            "description": description,
            "input": input_schema,
            "output": output,
            "type": "code",
            "handler": name,  # Handler name matches the tool name
            "func": func
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Registered handler: {name}")
        return func
    return decorator

@register_handler(
    name="fetch_url",
    description="Fetch content from a URL via GET request",
    input_schema={"url": "string"},
    output="string"
)
def fetch_url(config: Dict, params: Dict) -> str:
    url = params.get("url")
    if not url:
        raise ValueError("URL parameter is required")
    try:
        response = requests.get(url, timeout=config.get("timeout", 10))
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL {url}: {str(e)}")

@register_handler(
    name="fetch_text_url",
    description="Fetch URL content and extract readable text using BeautifulSoup",
    input_schema={"url": "string"},
    output="string"
)
def fetch_text_url(config: Dict, params: Dict) -> str:
    html_content = fetch_url(config, params)
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)

@register_handler(
    name="fetch_links_url",
    description="Fetch URL content and extract links with their text using BeautifulSoup",
    input_schema={"url": "string"},
    output="dict"
)
def fetch_links_url(config: Dict, params: Dict) -> Dict:
    html_content = fetch_url(config, params)
    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        link_text = a_tag.get_text(strip=True) or "Unnamed Link"
        links.append({"url": a_tag["href"], "text": link_text})
    return {"links": links}

@register_handler(
    name="pdf2text",
    description="Convert a PDF file to text given its file path",
    input_schema={"file_path": "string"},
    output="string"
)
def pdf2text(config: Dict, params: Dict) -> str:
    file_path = params.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Invalid or missing file path: {file_path}")
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to text: {str(e)}")

@register_handler(
    name="fetch_file",
    description="Fetch a file from a URL and store it temporarily, returning the file path",
    input_schema={"url": "string"},
    output="string"
)
def fetch_file(config: Dict, params: Dict) -> str:
    url = params.get("url")
    if not url:
        raise ValueError("URL parameter is required")
    storage_path = config.get("storage_path", tempfile.gettempdir())
    try:
        response = requests.get(url, timeout=config.get("timeout", 10), stream=True)
        response.raise_for_status()
        file_path = os.path.join(storage_path, tempfile.mktemp())
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch file from {url}: {str(e)}")

@register_handler(
    name="read_file",
    description="Read contents of a file with optional chunking",
    input_schema={"file_path": "string", "limit": "int", "start": "int", "end": "int"},
    output="string"
)
def read_file(config: Dict, params: Dict) -> str:
    file_path = params.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Invalid or missing file path: {file_path}")
    limit = params.get("limit")
    start = params.get("start", 0)
    end = params.get("end")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if start or end or limit:
            content = content[start:end if end else None]
            if limit and len(content) > limit:
                content = content[:limit]
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path}: {str(e)}")

@register_handler(
    name="write_file",
    description="Write content to a file and return its path, with optional file path override",
    input_schema={"file_content": "string", "file_path": "string"},
    output="string"
)
def write_file(config: Dict, params: Dict) -> str:
    content = params.get("file_content")
    if not content:
        raise ValueError("file_content parameter is required")
    file_path = params.get("file_path")
    storage_path = config.get("storage_path", tempfile.gettempdir())
    try:
        if not file_path:
            file_path = os.path.join(storage_path, tempfile.mktemp())
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(content))
        return file_path
    except Exception as e:
        raise RuntimeError(f"Failed to write file: {str(e)}")

@register_handler(
    name="run_snippet",
    description="Run Python code from content in a separate thread and return file path and output",
    input_schema={"code_content": "string"},
    output="dict"
)
def run_snippet(config: Dict, params: Dict) -> Dict:
    code_content = params.get("code_content")
    if not code_content:
        raise ValueError("code_content parameter is required")
    storage_path = config.get("storage_path", tempfile.gettempdir())
    file_path = os.path.join(storage_path, f"snippet_{os.urandom(4).hex()}.py")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_content)
    
    output = []
    error = []

    def run_code():
        try:
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                timeout=config.get("timeout", 10)
            )
            output.append(result.stdout)
            if result.stderr:
                error.append(result.stderr)
        except subprocess.TimeoutExpired as e:
            error.append(f"Execution timed out: {str(e)}")
        except Exception as e:
            error.append(f"Execution failed: {str(e)}")

    thread = threading.Thread(target=run_code)
    thread.start()
    thread.join()

    return {
        "file_path": file_path,
        "output": output[0] if output else "",
        "error": error[0] if error else None
    }

@register_handler(
    name="google_search",
    description="Perform a Google search and return results",
    input_schema={"query": "string", "limit": "int", "start": "int"},
    output="dict"
)
def google_search(config: Dict, params: Dict) -> Dict:
    query = params.get("query")
    if not query:
        raise ValueError("query parameter is required")
    limit = params.get("limit", config.get("default_limit", 10))
    start = params.get("start", 0)
    api_key = config.get("google_api_key")
    cse_id = config.get("google_cse_id")
    
    if api_key and cse_id:
        url = "https://www.googleapis.com/customsearch/v1"
        search_params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": min(limit, 10),
            "start": start
        }
        try:
            response = requests.get(url, params=search_params, timeout=config.get("timeout", 10))
            response.raise_for_status()
            data = response.json()
            return {"results": [{"title": item["title"], "url": item["link"]} for item in data.get("items", [])]}
        except requests.RequestException as e:
            raise RuntimeError(f"Google Custom Search failed: {str(e)}")
    else:
        try:
            results = list(search(query, num_results=limit, start=start))
            return {"results": [{"url": url} for url in results]}
        except Exception as e:
            raise RuntimeError(f"Google search failed: {str(e)}")

@register_handler(
    name="calculator",
    description="Evaluate a mathematical expression and return the result",
    input_schema={"expression": "string"},
    output="float"
)
def calculator(config: Dict, params: Dict) -> float:
    expression = params.get("expression")
    if not expression:
        raise ValueError("expression parameter is required")
    try:
        allowed_globals = {"__builtins__": {}}
        allowed_locals = {"pow": pow}
        result = eval(expression, allowed_globals, allowed_locals)
        if not isinstance(result, (int, float)):
            raise ValueError("Expression must evaluate to a number")
        return float(result)
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate expression '{expression}': {str(e)}")
