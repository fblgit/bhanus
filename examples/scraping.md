# Scraping Template Documentation

## Purpose
The `scraping.jinja2` template guides the model in summarizing scraped content from a URL. It ensures the output follows a consistent format, which is validated downstream in the scraping pipeline.

## Inputs
- `url` (string): The URL of the web page being summarized.
- `content` (string): The scraped content extracted from the URL.

## Expected Output
The model must produce a summary in this format:
```
Summary of <URL>: <summary text>
```
- `<URL>`: The exact URL provided in the input.
- `<summary text>`: A concise summary of the content, capturing its essence.

## Syntax and Formatting
- The output must begin with `Summary of ` followed by the URL and a colon (`:`).
- The summary should be brief (1-2 sentences) and focus on the main ideas.
- No additional text or deviations from the format are allowed.

## Few-Shot Examples

### Example 1
**Input:**
- `url`: "http://example.com"
- `content`: "This is an example website with sample content about technology trends in 2025."

**Output:**
```
Summary of http://example.com: An overview of technology trends expected in 2025.
```

### Example 2
**Input:**
- `url`: "http://news-site.com/article"
- `content`: "The article discusses the impact of AI on healthcare, highlighting advancements in diagnostics and patient care."

**Output:**
```
Summary of http://news-site.com/article: Explores AI's role in improving healthcare through better diagnostics and patient care.
```

### Example 3
**Input:**
- `url`: "http://blog.example.com/post"
- `content`: "A personal blog post reflecting on the author's experience with remote work during the pandemic."

**Output:**
```
Summary of http://blog.example.com/post: A personal reflection on remote work experiences during the pandemic.
```

### Example 4 (Edge Case)
**Input:**
- `url`: "http://empty.com"
- `content`: ""

**Output:**
```
Summary of http://empty.com: No content available to summarize.
