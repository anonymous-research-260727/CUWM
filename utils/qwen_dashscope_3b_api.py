import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Use environment variable or fallback to the known key if needed, 
# but adherence to user's file suggests os.getenv or specific key.
# I will use os.getenv as per user's file, but allow passing key.

def get_qwen_dashscope_3b_client():
    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("QWEN_API_KEY"))
         
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def get_qwen_dashscope_3b_response(
    messages,
    model="qwen2.5-vl-3b-instruct",
    temperature=0.0,
    top_p=0.8,
    max_tokens=2048,
    **kwargs
):
    """
    Get chat completion from Qwen 2.5 3B API via Dashscope.
    """
    client = get_qwen_dashscope_3b_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        return response
    except Exception as e:
        logger.error("Error calling Qwen 3B Dashscope API: %s", e)
        raise