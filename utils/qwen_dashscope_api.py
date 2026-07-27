import base64
import logging
import os
import time
from openai import OpenAI

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Constants for Qwen API
QWEN_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def get_qwen_api_client():
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )


def _file_to_base64_url(filepath):
    """Convert a local file path to a base64 data URL."""
    # Determine MIME type from extension
    ext = os.path.splitext(filepath)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(ext, 'image/png')
    
    with open(filepath, 'rb') as f:
        image_data = f.read()
    b64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{b64_data}"


def _ensure_valid_url(url_or_path):
    """Ensure the url is valid for API - convert local paths to base64."""
    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
        return url_or_path
    elif url_or_path.startswith('data:'):
        return url_or_path
    elif os.path.exists(url_or_path):
        # Local file path - convert to base64
        return _file_to_base64_url(url_or_path)
    else:
        # Unknown format, return as-is and let API handle error
        return url_or_path


def _convert_messages_for_dashscope(messages):
    """
    Convert messages from internal format to Dashscope API format.
    
    The Dashscope API expects:
    - 'image_url' type (not 'image')
    - 'image_url': {'url': '...'} format
    - URLs must be either http(s) URLs or base64 data URLs
    
    This handles common formats from internal code that might use:
    - 'image' type with 'image_url' key
    - Local file paths (converted to base64)
    """
    converted_messages = []
    for message in messages:
        if not isinstance(message.get("content"), list):
            # Plain text message, no conversion needed
            converted_messages.append(message)
            continue
        
        converted_content = []
        for item in message["content"]:
            item_type = item.get("type")
            
            if item_type == "text":
                converted_content.append(item)
            elif item_type == "image" or item_type == "image_url":
                # Convert 'image' to 'image_url' format for Dashscope
                # Handle various input formats
                if "image_url" in item:
                    url_data = item["image_url"]
                    if isinstance(url_data, dict):
                        url = url_data.get("url", "")
                    else:
                        url = url_data
                elif "url" in item:
                    url = item["url"]
                elif "image" in item:
                    url = item["image"]
                else:
                    # Skip malformed items
                    continue
                
                # Ensure the URL is valid for the API
                valid_url = _ensure_valid_url(url)
                
                converted_content.append({
                    "type": "image_url",
                    "image_url": {"url": valid_url}
                })
            else:
                # Keep other types as-is
                converted_content.append(item)
        
        converted_messages.append({
            "role": message["role"],
            "content": converted_content,
        })
    
    return converted_messages


def get_qwen_dashscope_api_response(
    messages,
    model="qwen3-vl-8b-instruct",
    temperature=0.0,
    top_p=0.8,
    max_tokens=2048,
    **kwargs
):
    """
    Get chat completion from Qwen API via Dashscope.
    
    Args:
        messages: List of messages in OpenAI format
        model: Model ID (default: qwen3-vl-8b-instruct)
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments to pass to client.chat.completions.create
    """
    client = get_qwen_api_client()
    
    # Convert messages to Dashscope format:
    # - 'image' type -> 'image_url' type
    # - Local file paths -> base64 data URLs
    converted_messages = _convert_messages_for_dashscope(messages)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=converted_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        return response
    except Exception as e:
        logger.error("Error calling Qwen Dashscope API: %s", e)
        raise
