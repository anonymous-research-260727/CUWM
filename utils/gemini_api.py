"""
Gemini API Wrapper

Provides an OpenAI-style API interface for Google Gemini models.
This allows plug-and-play usage with existing code that uses OpenAI-style APIs.

Usage:
    from utils.gemini_api import get_gemini_chat_completion, is_gemini_model
    
    # Use OpenAI-style messages
    response = get_gemini_chat_completion(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        model="gemini-2.5-flash"
    )
"""

from __future__ import annotations
import base64
import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from PIL import Image

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if GEMINI_API_KEY is set in environment


@dataclass
class GeminiChatMessage:
    """Mimics OpenAI ChatCompletionMessage structure"""
    content: str
    role: str = "assistant"
    # New field to store raw thought trace
    thought: Optional[str] = None


@dataclass
class GeminiChoice:
    """Mimics OpenAI Choice structure"""
    message: GeminiChatMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class GeminiChatCompletion:
    """Mimics OpenAI ChatCompletion structure"""
    choices: List[GeminiChoice]
    model: str
    id: str = "gemini-response"


class GeminiClient:
    """
    Client for Google Gemini models.
    Provides OpenAI-compatible interface for chat completions with vision.
    """
    
    _instance: Optional["GeminiClient"] = None
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Optional API key. If not provided, uses GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.debug("CWD: %s, Env vars: %s", os.getcwd(), list(os.environ.keys()))
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=self.api_key)
    
    @classmethod
    def get_instance(cls, api_key: Optional[str] = None) -> "GeminiClient":
        """Get or create a singleton instance."""
        if cls._instance is None:
            cls._instance = cls(api_key=api_key)
        return cls._instance
    
    @staticmethod
    def _decode_base64_image(data_url: str) -> Image.Image:
        """Decode a base64 data URL to PIL Image."""
        if "," in data_url:
            base64_data = data_url.split(",", 1)[1]
        else:
            base64_data = data_url
        
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    
    @staticmethod
    def _convert_openai_message_to_gemini(message: Dict[str, Any]) -> List[Any]:
        """
        Convert OpenAI-style message format to Gemini content parts.
        
        OpenAI format:
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]}
        
        Returns a list of content parts for Gemini.
        """
        content = message.get("content")
        
        if isinstance(content, str):
            return [content]
        
        if not isinstance(content, list):
            return [str(content)]
        
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url")
                
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = image_url
                
                if url.startswith("data:image"):
                    image = GeminiClient._decode_base64_image(url)
                    parts.append(image)
                elif os.path.exists(url):
                    image = Image.open(url).convert("RGB")
                    parts.append(image)
                else:
                    # URL - let Gemini handle it
                    parts.append(f"[Image from URL: {url}]")
            elif item.get("type") == "image":
                # Direct image path or PIL Image
                img = item.get("image")
                if isinstance(img, str) and os.path.exists(img):
                    image = Image.open(img).convert("RGB")
                    parts.append(image)
                elif isinstance(img, Image.Image):
                    parts.append(img)
                else:
                    parts.append(str(img))
        
        # DEBUG: Log parts types
        # print("DEBUG: Gemini parts types:", [type(p) for p in parts])
        return parts
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gemini-2.5-flash",
        temperature: Optional[float] = 0.1,
        max_tokens: int = 8192,
        **kwargs,
    ) -> GeminiChatCompletion:
        """
        Generate chat completion using Gemini.
        
        Args:
            messages: List of messages in OpenAI format
            model: Gemini model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
            
        Returns:
            GeminiChatCompletion object mimicking OpenAI response structure
        """
        # Convert messages to Gemini format
        # Gemini expects a flat list of content parts
        all_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            parts = self._convert_openai_message_to_gemini(msg)
            
            # Add role context for multi-turn
            if role == "system":
                all_parts.extend(["System instruction: "] + parts)
            elif role == "assistant":
                all_parts.extend(["Previous response: "] + parts)
            else:
                all_parts.extend(parts)
        
        # Configure generation
        config_kwargs = {"max_output_tokens": max_tokens}
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        # Handle thinking config - only enable when explicitly requested
        if kwargs.get("include_thoughts", False):
             config_kwargs["thinking_config"] = {"include_thoughts": True}
        
        config = types.GenerateContentConfig(
            safety_settings= [
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
            ],
            **config_kwargs
        )
        
        
        # DEBUG: Log configured safety settings
        # print("DEBUG: Safety Settings:", config.safety_settings)
        
        # Generate response
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=all_parts,
                config=config,
            )
        except Exception as e:
            logger.error("Gemini generate_content failed: %s", e)
            raise

        for candidate in response.candidates:
            if candidate.finish_reason != "STOP":
                logger.warning("Gemini finish reason: %s", candidate.finish_reason)
                if candidate.safety_ratings:
                    logger.warning("Safety ratings: %s", candidate.safety_ratings)
        
        # Extract text response and thoughts
        response_text = ""
        thought_trace = ""

        if response.parts:
            for part in response.parts:
                # Check for thought flag
                is_thought = False
                if hasattr(part, "thought") and part.thought:
                    is_thought = True
                
                # Extract text based on flag
                if hasattr(part, "text") and part.text:
                    if is_thought:
                        thought_trace += f"\n[Thought]: {part.text}\n"
                    else:
                        response_text += part.text
        
        # Fallback if no text found in parts but response.text exists (unlikely if parts exist)
        if not response_text and response.text:
            response_text = response.text
        
        if not response_text:
             logger.warning("Gemini response text is empty. Parts: %s", response.parts)
             
        # If response has candidates and candidates have content with parts, check deep
        if not thought_trace and response.candidates:
             cand = response.candidates[0]
             if hasattr(cand, "content") and cand.content and cand.content.parts:
                 for p in cand.content.parts:
                     if hasattr(p, "thought") and p.thought:
                         thought_trace += f"\n[Thought]: {p.thought}\n"

        return GeminiChatCompletion(
            choices=[
                GeminiChoice(
                    message=GeminiChatMessage(
                        content=response_text,
                        role="assistant",
                        thought=thought_trace if thought_trace else None
                    ),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model=model,
        )


def get_gemini_chat_completion(
    messages: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
    temperature: Optional[float] = 0.1,
    max_tokens: int = 8192,
    api_key: Optional[str] = None,
    **kwargs,
) -> GeminiChatCompletion:
    """
    Get chat completion from Gemini model.
    
    This is the main entry point that mimics OpenAI's get_chat_completion interface.
    
    Args:
        messages: List of messages in OpenAI format
        model: Gemini model name (default: gemini-2.5-flash)
        temperature: Sampling temperature (None uses model defaults)
        max_tokens: Maximum tokens to generate
        api_key: Optional API key (uses GEMINI_API_KEY env var if not provided)
        **kwargs: Additional generation arguments
        
    Returns:
        GeminiChatCompletion object with OpenAI-compatible structure
    """
    client = GeminiClient.get_instance(api_key=api_key)
    
    return client.chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def is_gemini_model(model_name: str) -> bool:
    """Check if the model name refers to a Gemini model."""
    return "gemini" in model_name.lower()


# Convenience alias
GeminiChatCompletionResponse = GeminiChatCompletion
