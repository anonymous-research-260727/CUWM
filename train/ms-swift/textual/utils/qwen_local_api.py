"""
Qwen3-VL Local Inference Wrapper

Provides an OpenAI-style API interface for Qwen3-VL models running locally via HuggingFace Transformers.
This allows plug-and-play usage with existing code that uses OpenAI-style APIs.

Usage:
    from utils.qwen_local_api import QwenVLClient, get_qwen_chat_completion
    
    # Initialize client (singleton pattern)
    client = QwenVLClient.get_instance(model_id="Qwen/Qwen3-VL-8B-Instruct")
    
    # Use OpenAI-style messages
    response = get_qwen_chat_completion(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        model="Qwen/Qwen3-VL-8B-Instruct"
    )
"""

from __future__ import annotations
import os
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import LogitsProcessor, LogitsProcessorList


class FrequencyPresencePenaltyLogitsProcessor(LogitsProcessor):
    """
    Applies frequency and presence penalties to logits during generation.
    
    Unlike the original implementation, this only penalizes tokens that appear
    in the *generated* portion, not the prompt (matching OpenAI behavior).
    """
    def __init__(self, frequency_penalty: float, presence_penalty: float, prompt_length: int):
        """
        Args:
            frequency_penalty: Penalty applied proportional to token frequency (0.0 = no penalty)
            presence_penalty: Penalty applied once per unique token (0.0 = no penalty)
            prompt_length: Number of tokens in the prompt (to exclude from counting)
        """
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.frequency_penalty == 0.0 and self.presence_penalty == 0.0:
            return scores
        
        # Only consider generated tokens (exclude prompt)
        generated_ids = input_ids[:, self.prompt_length:]
        
        # If no tokens generated yet, return unchanged scores
        if generated_ids.shape[1] == 0:
            return scores
            
        dtype = scores.dtype
        
        for i in range(generated_ids.shape[0]):  # Iterate over batch
            # Get unique tokens and counts from generated portion only
            unique_ids, counts = torch.unique(generated_ids[i], return_counts=True)
            
            # Apply frequency penalty (scales with count)
            if self.frequency_penalty != 0.0:
                scores[i, unique_ids] -= self.frequency_penalty * counts.to(dtype)
            
            # Apply presence penalty (flat penalty per unique token)
            if self.presence_penalty != 0.0:
                scores[i, unique_ids] -= self.presence_penalty
                 
        return scores


@dataclass
class QwenChatMessage:
    """Mimics OpenAI ChatCompletionMessage structure"""
    content: str
    role: str = "assistant"


@dataclass
class QwenChoice:
    """Mimics OpenAI Choice structure"""
    message: QwenChatMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class QwenChatCompletion:
    """Mimics OpenAI ChatCompletion structure"""
    choices: List[QwenChoice]
    model: str
    id: str = "qwen-response"
    

class QwenVLClient:
    """
    Singleton client for Qwen3-VL models.
    Provides OpenAI-compatible interface for chat completions with vision.
    """
    
    _instances: Dict[str, "QwenVLClient"] = {}
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        lora_path: str = None,
    ):
        """
        Initialize the Qwen3-VL model and processor.
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load model on ("auto", "cuda", "cuda:0", etc.)
            dtype: Model dtype (torch.bfloat16, torch.float16, etc.)
            use_flash_attention: Whether to use flash_attention_2 for efficiency
            lora_path: Path to LoRA adapter (optional)
        """
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        
        # Load model
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device,
            "trust_remote_code": True,
        }
        
        if use_flash_attention:
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_id, **load_kwargs
                )
            except Exception as e:
                print(f"Flash attention not available, falling back to default: {e}")
                del load_kwargs["attn_implementation"]
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_id, **load_kwargs
                )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id, **load_kwargs
            )
        
        # Load LoRA adapter if provided
        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model.load_adapter(lora_path)
        
        # Load processor
        # fix_mistral_regex=True fixes incorrect regex pattern in some tokenizers
        # See: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        
        print(f"Loaded Qwen VL local model: {model_id}")
    
    @classmethod
    def get_instance(
        cls,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        lora_path: str = None,
    ) -> "QwenVLClient":
        """
        Get or create a singleton instance for the specified model.
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load model on
            dtype: Model dtype
            use_flash_attention: Whether to use flash_attention_2
            lora_path: Path to LoRA adapter (optional)
            
        Returns:
            QwenVLClient instance
        """
        # Check if 'huggingface' subdirectory exists
        if os.path.isdir(os.path.join(model_id, "huggingface")):
            model_id = os.path.join(model_id, "huggingface")
            print(f"Detected 'huggingface' subdirectory, updated model_id to: {model_id}")

        key = f"{model_id}_{device}_{dtype}_{lora_path}"
        if key not in cls._instances:
            cls._instances[key] = cls(
                model_id=model_id,
                device=device,
                dtype=dtype,
                use_flash_attention=use_flash_attention,
                lora_path=lora_path,
            )
        return cls._instances[key]
    
    @staticmethod
    def _convert_openai_message_to_qwen(message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI-style message format to Qwen format.
        
        OpenAI format:
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]}
        
        Qwen format:
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image", "image": <PIL.Image or path>}
            ]}
        """
        if not isinstance(message.get("content"), list):
            # Plain text message, no conversion needed
            return message
        
        converted_content = []
        for item in message["content"]:
            if item.get("type") == "text":
                converted_content.append(item)
            elif item.get("type") == "image_url":
                # Handle image_url format
                image_url = item.get("image_url")
                
                # Handle both dict and string formats
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = image_url
                
                if url.startswith("data:image"):
                    # Base64 encoded image
                    image = QwenVLClient._decode_base64_image(url)
                    converted_content.append({
                        "type": "image",
                        "image": image,
                    })
                elif url.startswith("http://") or url.startswith("https://"):
                    # URL - Qwen can handle URLs directly
                    converted_content.append({
                        "type": "image",
                        "image": url,
                    })
                elif os.path.exists(url):
                    # Local file path
                    converted_content.append({
                        "type": "image",
                        "image": Image.open(url).convert("RGB"),
                    })
                else:
                    raise ValueError(f"Unknown image_url format: {url[:50]}...")
            elif item.get("type") == "image":
                # Already in Qwen format
                converted_content.append(item)
            else:
                # Unknown type, keep as is
                converted_content.append(item)
        
        return {
            "role": message["role"],
            "content": converted_content,
        }
    
    @staticmethod
    def _decode_base64_image(data_url: str) -> Image.Image:
        """Decode a base64 data URL to PIL Image."""
        import io
        
        # Extract base64 data from data URL
        # Format: data:image/png;base64,<data>
        if "," in data_url:
            base64_data = data_url.split(",", 1)[1]
        else:
            base64_data = data_url
        
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    
    @torch.inference_mode()
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: int = 3407,
        do_sample: bool = False,
        **kwargs,
    ) -> QwenChatCompletion:
        """
        Generate chat completion using Qwen3-VL.
        
        Args:
            messages: List of messages in OpenAI format
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty (1.0 = no penalty)
            frequency_penalty: OpenAI-style frequency penalty (0.0 = no penalty)
            presence_penalty: OpenAI-style presence penalty (0.0 = no penalty)
            seed: Random seed for reproducibility
            do_sample: Whether to use sampling (False for greedy)
            **kwargs: Additional generation arguments
            
        Returns:
            QwenChatCompletion object mimicking OpenAI response structure
        """
        # Convert messages to Qwen format
        qwen_messages = [
            self._convert_openai_message_to_qwen(msg) for msg in messages
        ]
        
        # Prepare inputs using processor
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Get prompt length for penalty processor
        prompt_length = inputs.input_ids.shape[1]
        
        # Filter out sampling params from kwargs to avoid passing them when do_sample=False
        sampling_params = {"temperature", "top_p", "top_k"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in sampling_params}
        
        # Generation parameters - base params only
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            **filtered_kwargs,
        }
        
        # Set up logits processors for frequency/presence penalties
        logits_processor = LogitsProcessorList()
        if presence_penalty != 0.0 or frequency_penalty != 0.0:
            logits_processor.append(
                FrequencyPresencePenaltyLogitsProcessor(
                    frequency_penalty=frequency_penalty, 
                    presence_penalty=presence_penalty,
                    prompt_length=prompt_length,
                )
            )
            gen_kwargs["logits_processor"] = logits_processor

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Only add sampling parameters when actually sampling
        if do_sample:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        
        # Generate
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # Return OpenAI-style response
        return QwenChatCompletion(
            choices=[
                QwenChoice(
                    message=QwenChatMessage(
                        content=output_text[0] if output_text else "",
                        role="assistant",
                    ),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model=self.model_id,
        )


# Global client cache for convenience
_qwen_clients: Dict[str, QwenVLClient] = {}


def get_qwen_chat_completion(
    messages: List[Dict[str, Any]],
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
    max_new_tokens: int = 2048,
    temperature: Optional[float] = 0.0,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int = 3407,
    use_flash_attention: bool = True,
    lora_path: str = None,
    **kwargs,
) -> QwenChatCompletion:
    """
    Get chat completion from Qwen3-VL model.
    
    This is the main entry point that mimics OpenAI's get_chat_completion interface.
    
    Args:
        messages: List of messages in OpenAI format
        model: Model ID (HuggingFace model ID or local path)
        device: Device to run on ("auto", "cuda", "cuda:0", etc.)
        dtype: Model dtype
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy, >0 enables sampling)
        top_p: Nucleus sampling probability (only used when temperature > 0)
        top_k: Top-k sampling (only used when temperature > 0)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        frequency_penalty: OpenAI-style frequency penalty (0.0 = no penalty)
        presence_penalty: OpenAI-style presence penalty (0.0 = no penalty)
        seed: Random seed for reproducibility
        use_flash_attention: Whether to use flash_attention_2
        lora_path: Path to LoRA adapter (optional)
        **kwargs: Additional generation arguments
        
    Returns:
        QwenChatCompletion object with OpenAI-compatible structure
    """
    # Get or create client instance
    client = QwenVLClient.get_instance(
        model_id=model,
        device=device,
        dtype=dtype,
        use_flash_attention=use_flash_attention,
        lora_path=lora_path,
    )
    
    # Determine if we should sample based on temperature
    do_sample = temperature is not None and temperature > 0
    
    # Prepare generation kwargs - don't include sampling params here,
    # let chat_completion handle them based on do_sample
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "do_sample": do_sample,
        **kwargs,
    }

    # Apply specific settings for the merged SFT model (checkpoint-450-merged)
    # Use substring matching for robustness against different path formats
    # Apply specific settings for the merged SFT model (checkpoint-450-merged)
    # Use substring matching for robustness against different path formats
    if "checkpoint-450-merged" in model or "grpo" in model.lower():
        gen_kwargs["presence_penalty"] = 0.6
        gen_kwargs["frequency_penalty"] = 0.6
        print(f"[QwenLocal] Detected checkpoint-450-merged/grpo model, applying penalties: "
              f"presence={gen_kwargs['presence_penalty']}, frequency={gen_kwargs['frequency_penalty']}")
    
    # Only pass sampling params when sampling is enabled
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        # Greedy decoding (temperature=0)
        print(f"[QwenLocal] Using greedy decoding (temperature=0, do_sample=False)")

    return client.chat_completion(messages, **gen_kwargs)


def is_qwen_model(model_name: str) -> bool:
    """Check if the model name refers to a Qwen VL model."""
    model_lower = model_name.lower()
    return "qwen" in model_lower and ("vl" in model_lower or "vision" in model_lower)


# Convenience alias for drop-in replacement
QwenChatCompletionResponse = QwenChatCompletion