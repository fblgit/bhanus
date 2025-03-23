## Register a code-based tool
```
{
    "jsonrpc": "2.0",
    "method": "register_tool",
    "params": {
        "name": "fetch_data",
        "description": "Fetches data from an API",
        "type": "code",
        "handler": "api_call",
        "config": {"url": "https://api.example.com", "method": "GET"},
        "input": {"query": "string"},
        "output": "dict"
    },
    "id": 1
}
```
## Invoke the tool
```
{
    "jsonrpc": "2.0",
    "method": "invoke_tool",
    "params": {
        "tool_name": "fetch_data",
        "query": "test",
        "consent": true
    },
    "id": 2
}
```
