#!/usr/bin/env python3
"""Gemini-based image generation for UI prediction."""

from google import genai
from google.genai import types
from PIL import Image
import os
import time
import re
import logging
from dotenv import load_dotenv

load_dotenv()

def generate_image(image, prompt, size=None, model_name="banana", max_retries=5, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    # Handle image as list or single image
    input_image = image[0] if isinstance(image, list) else image

    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    client = genai.Client(api_key=api_key)
    full_prompt = prompt

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=["Input Image:", input_image, full_prompt],
                config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
            )

            for part in response.parts:
                if part.inline_data is not None:
                    return part.as_image()

            raise ValueError("No image in API response")

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    match = re.search(r'retry in ([\d.]+)s', str(e))
                    delay = float(match.group(1)) + 1 if match else 60.0
                    logger.warning(f"Rate limited, waiting {delay:.1f}s")
                    time.sleep(delay)
                    continue
            raise
