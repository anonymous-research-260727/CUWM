
import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from utils.cloudgpt_aoai import get_chat_completion, encode_image, get_openai_client
from utils.qwen_local_api import get_qwen_chat_completion, is_qwen_model
from utils.gemini_api import get_gemini_chat_completion, is_gemini_model
from utils.qwen_dashscope_api import get_qwen_dashscope_api_response
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from multiprocessing import Pool


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out!")


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=300):
    """
    Run a function with a timeout. If the function doesn't complete within
    timeout_seconds, raise TimeoutError.
    
    Args:
        func: Function to run
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        timeout_seconds: Timeout in seconds (default 5 minutes = 300 seconds)
    
    Returns:
        Result of func(*args, **kwargs)
    
    Raises:
        TimeoutError: If function doesn't complete within timeout
    """
    if kwargs is None:
        kwargs = {}
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


def setup_logger(log_file):
    """Setup logger that logs to both console and file"""
    logger = logging.getLogger('AgentLogger')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicate logs
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class Agent:
    """
    - model_key == "base": 使用 diffusers 的 Qwen/Qwen-Image-Edit-2509
    - model_key != "base": 使用 diffsynth 的 QwenImagePipeline 并加载对应 epoch-xx.safetensors
    初始化后只固定使用一个模型，不做热切换。
    """

    BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
    # 你把 LoRA 权重都放这里，文件名示例：epoch-10.safetensors
    LORA_DIR = os.environ.get("LORA_DIR", "models/lora")
    ROOT = os.environ.get("DATA_ROOT", "data")

    def __init__(self, args, logger):
        self.logger = logger
        self.logger.info(f"Initializing Agent with args={args}...")
        self.model_key = args.model_key
        self.device = args.device
        self.exp_name = args.exp_name
        self.dtype = torch.bfloat16
        self.pipe = None
        self.mode = args.mode
        self.prompt_version = getattr(args, 'prompt_version', 'v1')
        self.client = get_openai_client()
        self.qwen_lora_path = getattr(args, 'qwen_lora_path', None)
    
    def load_model(self, device: str = None):
        if self.model_key == "base":
            self.logger.info(f"Loading base model from {self.BASE_MODEL_ID}...")
            from diffusers import DiffusionPipeline
            self.pipe = DiffusionPipeline.from_pretrained(
                self.BASE_MODEL_ID,
                torch_dtype=self.dtype,
            ).to(device if device else self.device)
        elif self.model_key == "epoch-24":
            from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
            from config.configs import MODEL_CONFIGS, PROCESSOR_CONFIG
            # diffsynth pipeline
            self.pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=self.dtype,
                device=device if device else self.device,
                model_configs=MODEL_CONFIGS,
                tokenizer_config=None,
                processor_config=PROCESSOR_CONFIG,
            )

            lora_path = os.path.join(self.LORA_DIR, f"{self.model_key}.safetensors")
            print(lora_path)
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA not found: {lora_path}")
            self.pipe.load_lora(self.pipe.dit, lora_path)
        else:
            self.pipe = self.model_key
            
    def _get_input_path(self, folder, filename: str):
        return os.path.join(self.ROOT, folder, filename)
            
    def _get_output_path(self, folder, filename: str):
        rel_folder = os.path.relpath(folder, self.ROOT)
        save_folder = os.path.join(self.ROOT, "agent_eval_result", self.exp_name, rel_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        return os.path.join(save_folder, filename)
            
    def _generate_option_generation_prompt(
        self, 
        num_options: int,
        app: str,
        folder: str,
    ):
        save_prompt_path = self._get_output_path(folder, "option_generation_prompt.txt")      
        from eval.prompts import (
            ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT,
            ACTION_PREDICTION_A11Y_USER_PROMPT_GPT,
            SUPPORTED_ACTIONS,
        )
        
        supported_actions = SUPPORTED_ACTIONS[app]
        sys_prompt = ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT.replace("{num_options}", str(num_options))
        
        with open(self._get_input_path(folder, "request.txt"), "r", encoding="utf-8") as f:
            instruction = f.read().strip()
        
        with open(self._get_input_path(folder, "a11y.json"), "r", encoding="utf-8") as f:
            a11y = json.load(f)
            
        # print(a11y)
        
        usr_prompt = ACTION_PREDICTION_A11Y_USER_PROMPT_GPT.format(
            instruction=instruction,
            a11y=json.dumps(a11y, indent=2),
            actions=supported_actions,
            num_options=num_options,
        )
        
        prompt = sys_prompt + "\n\n" + usr_prompt
            
        with open(save_prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
            # self.logger.info(f"Option generation prompt saved to {save_prompt_path}")
            
        return prompt
    
        
    def _generate_action_selection_prompt(
        self,
        action_outcomes: list[dict],
        request: str,
        folder: str,
    ):
        from eval.prompts import (
            GUI_ACTION_SELECTION_PROMPT_1,
            GUI_ACTION_SELECTION_PROMPT_DEFAULT,
            GUI_ACTION_SELECTION_PROMPT2,
            GUI_ACTION_SELECTION_PROMPT3,
        )
        # === 1. 构造 content 数组（text / image 交替） ===
        content, prompt, image_paths = [], "", []

        # ---- Prompt 头部（text）----
        
        # Use different header for v3 (predictive) if needed, or keep generic one
        if self.prompt_version == "v3":
             from eval.prompts import GUI_ACTION_SELECTION_PROMPT_PREDICTIVE
             header_prompt = GUI_ACTION_SELECTION_PROMPT_PREDICTIVE.format(request=request)
        else:
             header_prompt = GUI_ACTION_SELECTION_PROMPT_1.format(request=request)
             
        content.append({"type": "text", "text": header_prompt})
        prompt += header_prompt + "\n\n"
        
        prev_img_path = self._get_input_path(folder, "prev.png")
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(prev_img_path)},
        })
        prompt += f"[Image: {prev_img_path}]\n\n"
        image_paths.append(prev_img_path)
        
        if self.prompt_version == "default" or self.prompt_version == "v1":
             prompt_dict = GUI_ACTION_SELECTION_PROMPT_DEFAULT
        elif self.prompt_version == "v2":
             prompt_dict = GUI_ACTION_SELECTION_PROMPT2
        elif self.prompt_version == "v3":
             prompt_dict = GUI_ACTION_SELECTION_PROMPT3
        else:
             print(f"Warning: Unknown prompt version {self.prompt_version}, using default (v1)")
             prompt_dict = GUI_ACTION_SELECTION_PROMPT_DEFAULT
        
        try:
             header_prompt_2 = prompt_dict[self.mode].format(request=request)
        except Exception:
             # Fallback if the prompt doesn't need formatting or request is missing
             header_prompt_2 = prompt_dict[self.mode]
             
        content.append({
            "type": "text",
            "text": header_prompt_2,
        })
        prompt += header_prompt_2 + "\n\n"
        

        # ---- Action options（text + 可选 image）----
        for idx, ele in enumerate(action_outcomes):
            if self.prompt_version == "v3":
                # Predictive phrasing for v3
                # "If you proceed with this action {action}, then this {prediction} is what will happen"
                
                # Format action nicely
                action_str = f"{ele['action']}"
                
                option_text = f"If you proceed with this action {action_str}, then "
                
                if self.mode in ("text", "text+image"):
                    option_text += f"this text prediction '{ele['text_synthesis']}' "
                
                if self.mode in ("image", "text+image"):
                     if self.mode == "text+image":
                         option_text += "and "
                     option_text += "this Visual GUI state (see below) "
                
                option_text += "is what will happen.\n"
                
            else:
                # Standard phrasing
                option_text = (
                    f"Action Option {idx + 1}:\n"
                    f" - Action: {ele['action']}\n"
                )

                if self.mode in ("text", "text+image"):
                    option_text += f" - Predicted State Text: {ele['text_synthesis']}\n"
                
                if self.mode in ("image", "text+image"):
                    option_text += f" - Predicted State Image: see below.\n"

            content.append({
                "type": "text",
                "text": option_text,
            })
            prompt += option_text + "\n"

            if self.mode in ("image", "text+image"):
                img_path = ele["image_synthesis"]
                content.append({
                    "type": "image_url",
                    "image_url": {"url": encode_image(img_path)},
                })
                prompt += f"[Image: {img_path}]\n\n"
                image_paths.append(img_path)

        # ---- 收尾指令（text）----
        end_text = "The argument specifications for the action options are described below:\n"
        from prompts.gui_descriptions import GUI_DESCRIPTION
        app = self._get_app(folder)
        actions = set([ele["action"]["function"] for ele in action_outcomes])
        
        for action in actions:
            if action in GUI_DESCRIPTION[app]:
                end_text += GUI_DESCRIPTION[app][action] + "\n"
        
        
        end_text += f"Now, analyze all options and return only the selected 'action_idx' and a short 'thought' in JSON format. 'action_idx' MUST be an integer between 1 and {len(action_outcomes)} (inclusive). Do NOT output any text outside the JSON object."
        content.append({
            "type": "text",
            "text": end_text,
        })
        prompt += end_text + "\n"
        

        # === 2. 返回 API 需要的 message ===
        message = {
            "role": "user",
            "content": content,
        }

        return [message], prompt, image_paths

    @torch.inference_mode()
    def _generate_next_image(
        self,
        edit_path: str,
        prompt_path: str | None = None,
        prompt: str | None = None,
        output_path: str | None = None,
        num_inference_steps: int = 40,
        seed: int = 123,
        device: str = None,
    ):
        if self.pipe is None:
            self.load_model(device=device)
        if prompt is None:
            if prompt_path is None:
                raise ValueError("Need prompt or prompt_path.")
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

        edit_image = Image.open(edit_path).convert("RGB")
        h, w = edit_image.size[1], edit_image.size[0]

        if self.model_key == "base":
            out = self.pipe(
                image=edit_image,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=h,
                width=w,
            ).images[0]
        elif self.model_key in ["dall-e-3", "gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"]:
            from utils.call_image_gen_api import generate_image
            out = generate_image(
                image_paths=[edit_path],
                prompt=prompt,
                size=f"{w}x{h}",
                model_name=self.model_key,
            )
        elif self.model_key == "banana":
            from utils.banana_image_gen import generate_image
            out = generate_image(
                image=[edit_image],
                prompt=prompt,
            )    
        else:
            out = self.pipe(
                prompt,
                edit_image=edit_image,
                seed=seed,
                num_inference_steps=num_inference_steps,
                height=h,
                width=w,
            )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out.save(output_path)
        return out
    
    def _generate_llm_response(
        self,
        prompt: str = None,
        image_paths: list[str] = None,
        message: list[dict] = None,
        engine: str = "gpt-5.2-chat-20251211",
        temperature: float = None,
        include_thoughts: bool = False,
        **kwargs,
    ):
        """
        Generate LLM response given a text prompt and optional images.
        
        Supports OpenAI-style models (GPT), Qwen VL models, and Gemini models.
        - Qwen models are detected by checking if model name contains 'qwen' and 'vl'.
        - Gemini models are detected by checking if model name contains 'gemini'.
        """
        assert prompt is not None or message is not None, "Either prompt or message must be provided."
        
        # Check which model type to use
        # Treat paths starting with / as local Qwen models
        use_qwen = is_qwen_model(engine) or engine.startswith("/")
        use_gemini = is_gemini_model(engine)
        
        if message:
            messages = message
        else:
            if image_paths is None:
                image_paths = []
            
            if type(image_paths) is str:
                image_paths = [image_paths]
            
            # Build content list
            content = [{"type": "text", "text": prompt}]
            
            # Add images if provided
            for img_path in image_paths:
                if use_qwen or use_gemini:
                    # Qwen and Gemini use direct image paths
                    content.append({
                        "type": "image",
                        "image": img_path,
                    })
                else:
                    # OpenAI uses base64 encoded images
                    image_url = encode_image(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        
        if use_gemini:
            # Use Gemini API
            kwargs = {"model": engine}
            if temperature is not None:
                kwargs["temperature"] = temperature
            
            if include_thoughts:
                kwargs["include_thoughts"] = True
                
            response = get_gemini_chat_completion(
                messages=messages,
                **kwargs,
            )
        elif "qwen2.5-vl-3b-instruct" in engine.lower():
             # Use Qwen 2.5 3B API
             from utils.qwen_dashscope_3b_api import get_qwen_dashscope_3b_response
             qwen_kwargs = {"model": engine}
             if temperature is not None:
                qwen_kwargs["temperature"] = temperature
             qwen_kwargs.update(kwargs)
             
             # Filter out unsupported arguments for OpenAI-compatible API
             if "repetition_penalty" in qwen_kwargs:
                 qwen_kwargs.pop("repetition_penalty")
             
             response = get_qwen_dashscope_3b_response(
                messages=messages,
                **qwen_kwargs
             )
        elif "qwen2.5-vl-7b-instruct" in engine.lower():
             # Use Qwen 2.5 7B API
             from utils.qwen_dashscope_7b_api import get_qwen_dashscope_7b_response
             qwen_kwargs = {"model": engine}
             if temperature is not None:
                qwen_kwargs["temperature"] = temperature
             qwen_kwargs.update(kwargs)
             
             # Filter out unsupported arguments for OpenAI-compatible API
             if "repetition_penalty" in qwen_kwargs:
                 qwen_kwargs.pop("repetition_penalty")
             
             response = get_qwen_dashscope_7b_response(
                messages=messages,
                **qwen_kwargs
             )
        elif use_qwen:
            # Check if we should use the Qwen API (Dashscope) or local model
            if "qwen3-vl-8b-instruct" in engine.lower():
                # Use Qwen3 VL 8B API
                qwen_kwargs = {"model": engine}
                
                # FORCE temperature to 0 (or very low) as requested
                qwen_kwargs["temperature"] = 0.001
                
                # Add extra kwargs (but we overwrote temperature)
                qwen_kwargs.update(kwargs)
                # Re-enforce temperature
                qwen_kwargs["temperature"] = 0.001
                
                # Remove repetition_penalty as Dashscope API doesn't support it
                qwen_kwargs.pop("repetition_penalty", None)
                
                response = get_qwen_dashscope_api_response(
                    messages=messages,
                    **qwen_kwargs
                )
                # Fall through to standard extraction logic at the end of function
            else:
                # Use local Qwen VL API
                qwen_kwargs = {"model": engine}
                if temperature is not None:
                    qwen_kwargs["temperature"] = temperature
                
                # Add default repetition_penalty if not specified
                if "repetition_penalty" not in kwargs:
                    qwen_kwargs["repetition_penalty"] = 1.05
                
                # Add extra kwargs (repetition_penalty, etc.)
                qwen_kwargs.update(kwargs)
                
                response = get_qwen_chat_completion(
                    messages=messages,
                    device=self.device,
                    **qwen_kwargs,
                )
        else:
            # Use OpenAI API
            if temperature is not None:
                response = get_chat_completion(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                response = get_chat_completion(
                    model=engine,
                    messages=messages,
                )
        
        # Extract text
        response_text = response.choices[0].message.content
        
        if include_thoughts:
            thought = getattr(response.choices[0].message, "thought", None)
            return response_text, thought
            
        return response_text

    def _get_text_synthesis(
        self,
        folder: str,
        action_idx: int,
        action: dict,
        llm_model: str = "gpt-5.2-chat-20251211",
    ):
        # self.logger.info(f"Generating text synthesis for action {action_idx}...")
        save_path = self._get_output_path(folder, f"text_synthesis_action_{action_idx}.txt")
        prompt_path = self._get_output_path(folder, f"text_synthesis_prompt_action_{action_idx}.txt")
        
        if os.path.exists(save_path):
            self.logger.info(f"Text synthesis already exists at {save_path}, loading...")
            with open(save_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        from scripts.generate_text_pred_prompt import generate_nl_description
        content, prompt = generate_nl_description(
            folder,
            action=action,
            client=self.client,
            llm_model=llm_model,
            temperature=0,
            lora_path=self.qwen_lora_path,
        )
            
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        self.logger.info(f"Text synthesis saved to {save_path}")
        return content

    def _get_image_synthesis(
        self,
        folder: str,
        action_idx: int,
        text_synthesis: str,
        device: str = None,
    ):
        save_path = self._get_output_path(
            os.path.join(folder, f"generated_{self.model_key}"),
            f"image_synthesis_action_{action_idx}.png"
        )
        if os.path.exists(save_path):
            # self.logger.info(f"Image synthesis already exists at {save_path}, loading...")
            return save_path
        
        prompt = text_synthesis
        edit_image_path = self._get_input_path(folder, "prev.png")
        self._generate_next_image(
            edit_path=edit_image_path,
            prompt=prompt,
            output_path=save_path,
            num_inference_steps=40,
            seed=12345,
            device=device,
        )
        self.logger.info(f"Image synthesis saved to {save_path}")
        return save_path
    
    def _get_app(self, folder):
        if "word" in folder.lower():
            app = "word"
        elif "excel" in folder.lower():
            app = "excel"
        elif "ppt" in folder.lower():
            app = "ppt"
        else:
            raise ValueError(f"Cannot infer app type from folder: {folder}")
        return app
    
    def _normalize_action(self, action):
        """Normalize an action to a comparable format for deduplication."""
        normalized = {
            "function": action.get("function", ""),
            "args": json.dumps(action.get("args", {}), sort_keys=True),
            "status": action.get("status", "")
        }
        return json.dumps(normalized, sort_keys=True)
    
    def _has_duplicate_actions(self, action_options):
        """Check if there are duplicate actions in the list."""
        normalized_actions = [self._normalize_action(action) for action in action_options]
        return len(normalized_actions) != len(set(normalized_actions))
    
    def action_option_generation(
        self, 
        folder: str,
        num_options: int = 5,
        llm_model: str = "gpt-5.2-chat-20251211",
    ):
        save_path = self._get_output_path(folder, "action_options.json")
        raw_save_path = self._get_output_path(folder, "action_options_raw.json")
        if os.path.exists(save_path):
            # self.logger.info(f"Action options already exist at {save_path}, loading...")
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)
            
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                app = self._get_app(folder)
                option_generation_prompt = self._generate_option_generation_prompt(
                    num_options=num_options,
                    app=app,
                    folder=folder,
                )
                image_path = self._get_input_path(folder, "prev.png")
                annotated_image_path = self._get_input_path(folder, "prev_annotated.png")
                # Use temperature 0 for all models for deterministic output
                temperature = 0
                # Use repetition penalty to reduce duplicates
                repetition_penalty = 1.0 if (is_qwen_model(llm_model) or llm_model.startswith("/")) else None
                
                # Enable thoughts for Gemini models
                use_gemini = is_gemini_model(llm_model)
                include_thoughts = use_gemini
                
                llm_result = self._generate_llm_response(
                    prompt=option_generation_prompt,
                    image_paths=[image_path, annotated_image_path],
                    engine=llm_model,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    include_thoughts=include_thoughts,
                )
                
                if include_thoughts:
                    response, thought = llm_result
                    if thought:
                        thought_save_path = self._get_output_path(folder, "action_options_thought.txt")
                        with open(thought_save_path, "w", encoding="utf-8") as f:
                            f.write(thought)
                        self.logger.info(f"Action options thought saved to {thought_save_path}")
                else:
                    response = llm_result
                # self.logger.info(f"LLM response: {response}")
                
                # jsonify and save
                import re
                # Find the first JSON list
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    response_json_str = match.group(0)
                else:
                    response_json_str = response.replace("```json", "").replace("```", "").strip()
                
                try:
                    response_json = json.loads(response_json_str)
                except json.JSONDecodeError:
                    try:
                        import ast
                        response_json = ast.literal_eval(response_json_str)
                    except Exception:
                        self.logger.error(f"Failed to parse JSON list with json and ast: {response}")
                        raise
                
                # Validate: should be a list with num_options elements, each being a dict
                if not isinstance(response_json, list):
                    raise ValueError(f"Expected a list, got {type(response_json)}")
                if len(response_json) != num_options:
                    raise ValueError(f"Expected {num_options} options, got {len(response_json)}")
                if not all(isinstance(item, dict) for item in response_json):
                    raise ValueError("All elements should be dictionaries")

                
                with open(raw_save_path, "w", encoding="utf-8") as f:
                    json.dump(response_json, f, indent=2)
                    # self.logger.info(f"Action options raw saved to {save_path}")
                
                # process actions
                action_options = []
                for ele in response_json:
                    action_option = ele["tool_call"]
                    if "control_label" in action_option["args"]:
                        control_label = action_option["args"]["control_label"]
                        with open(self._get_input_path(folder, "a11y.json"), "r", encoding="utf-8") as f:
                            a11y = json.load(f)
                            if type(control_label) == int and 1 <= control_label <= len(a11y):
                                control_info = a11y[control_label - 1]
                                assert control_info["label"] == control_label
                                del control_info["control_rect"]
                                del control_info["source"]
                                del control_info["label"]
                                action_option["args"]["control_info"] = control_info
                        del action_option["args"]["control_label"]
                    action_options.append(action_option)

                # Log warning if duplicate actions exist but don't raise error
                if self._has_duplicate_actions(action_options):
                    duplicate_indices = []
                    normalized_actions = {}
                    for idx, action in enumerate(action_options):
                        norm_action = self._normalize_action(action)
                        if norm_action in normalized_actions:
                            duplicate_indices.append((normalized_actions[norm_action], idx))
                        else:
                            normalized_actions[norm_action] = idx

                    self.logger.warning(
                        f"Duplicate actions found at indices: {duplicate_indices}. "
                        f"Saving anyway - will regenerate later."
                    )

                    # Save duplicate information to a separate file
                    duplicate_info_path = self._get_output_path(folder, "duplicate_actions.json")
                    duplicate_info = {
                        "has_duplicates": True,
                        "duplicate_indices": duplicate_indices,
                        "folder": folder,
                    }
                    with open(duplicate_info_path, "w", encoding="utf-8") as f:
                        json.dump(duplicate_info, f, indent=2)
                    self.logger.info(f"Duplicate info saved to {duplicate_info_path}")

                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(action_options, f, indent=2)
                    self.logger.info(f"Action options saved to {save_path}")
                return response_json

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    self.logger.error(f"All {max_retries} attempts failed.")
                    pass
    
    
    def action_outcome_synthesis(
        self,
        mode: str,
        folder: str,
        num_options: int = 5,
        llm_model: str = "gpt-5.2-chat-20251211",
        device: str = None,
    ):
        assert mode in ["text", "image", "text+image", "none"]
        if mode == "image" or mode == "text+image":
            save_path = self._get_output_path(
                folder=os.path.join(folder, f"generated_{self.model_key}"),
                filename=f"action_outcome_synthesis_{mode}.json"
            )
        else:
            save_path = self._get_output_path(folder, f"action_outcome_synthesis_{mode}.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)
            
        action_option_filepath = self._get_output_path(folder, "action_options.json")
        with open(action_option_filepath, "r", encoding="utf-8") as f:
            action_options = json.load(f)
            
        result = []
        
        for action_idx, action_option in enumerate(action_options):
            self.logger.info(f"Processing action option {action_idx}: {action_option}...")
            action = action_option
            
            if mode == "none":
                result.append({
                    "action": action,
                })
            elif mode == "text":
                text_synthesis = self._get_text_synthesis(
                    folder=folder,
                    action_idx=action_idx,
                    action=action,
                    llm_model=llm_model,
                )
                result.append({
                    "action": action,
                    "text_synthesis": text_synthesis,
                })
            elif mode == "image" or mode == "text+image":
                text_synthesis = self._get_text_synthesis(
                    folder=folder,
                    action_idx=action_idx,
                    action=action,
                    llm_model=llm_model,
                )
                image_synthesis = self._get_image_synthesis(
                    folder=folder,
                    action_idx=action_idx,
                    text_synthesis=text_synthesis,
                    device=device,
                )
                if mode == "image":
                    result.append({
                        "action": action,
                        "image_synthesis": image_synthesis,
                    })
                elif mode == "text+image":
                    result.append({
                        "action": action,
                        "text_synthesis": text_synthesis,
                        "image_synthesis": image_synthesis,
                    })
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        return result        
    
    def action_selection(self, folder: str, llm_model: str = "gpt-5.2-chat-20251211"):
        if self.mode == "image" or self.mode == "text+image":
            save_folder = os.path.join(folder, f"generated_{self.model_key}")
        else:
            save_folder = folder
            
        save_path = self._get_output_path(save_folder, f"action_selection_{self.mode}.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)
            
        action_outcome_filepath = self._get_output_path(save_folder, f"action_outcome_synthesis_{self.mode}.json")
        with open(action_outcome_filepath, "r", encoding="utf-8") as f:
            action_outcomes = json.load(f)
        
        request_filepath = self._get_input_path(folder, "request.txt")
        with open(request_filepath, "r", encoding="utf-8") as f:
            request = f.read().strip()
            
        message, selection_prompt, image_paths = self._generate_action_selection_prompt(
            action_outcomes=action_outcomes, 
            request=request,
            folder=folder,
        )
        
        save_message_path = self._get_output_path(save_folder, f"action_selection_prompt_message_{self.mode}.json")
        with open(save_message_path, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=2)
        
        save_prompt_path = self._get_output_path(save_folder, f"action_selection_prompt_text_{self.mode}.txt")
        save_image_path = self._get_output_path(save_folder, f"action_selection_prompt_images_{self.mode}.json")
        
        with open(save_prompt_path, "w", encoding="utf-8") as f:
            f.write(selection_prompt)
            # self.logger.info(f"Selection prompt text saved to {save_prompt_path}")
        #
        with open(save_image_path, "w", encoding="utf-8") as f:
            json.dump(image_paths, f, indent=2)
            self.logger.info(f"Selection prompt images saved to {save_image_path}")
    
        # Clean up stale thought file before regeneration
        thought_save_path = self._get_output_path(save_folder, f"action_selection_response_thought_{self.mode}.txt")
        if os.path.exists(thought_save_path):
            try:
                os.remove(thought_save_path)
            except OSError:
                pass

        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Use temperature 0 for all models for deterministic output
                temperature = 0
                
                # Enable thoughts for Gemini models to capture reasoning
                use_gemini = is_gemini_model(llm_model)
                include_thoughts = use_gemini
                
                llm_result = self._generate_llm_response(
                    message=message,
                    engine=llm_model,
                    temperature=temperature,
                    include_thoughts=include_thoughts,
                )
                
                # Handle tuple return when thoughts are included
                if include_thoughts:
                    response, thought = llm_result
                    # Save thought to file if present
                    if thought:
                        with open(thought_save_path, "w", encoding="utf-8") as f:
                            f.write(thought)
                        self.logger.info(f"Thought saved to {thought_save_path}")
                else:
                    response = llm_result
                
                # Save raw response for debugging
                raw_response_path = save_path.replace(".json", "_raw_response.txt")
                with open(raw_response_path, "w", encoding="utf-8") as f:
                    f.write(response)
                self.logger.info(f"Raw LLM response saved to {raw_response_path}")
                
                # Robust JSON Parsing Strategy
                import re
                import ast
                
                response_json = None
                
                # Method 1: Extraction from Code Blocks (High Confidence)
                code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
                if code_block_match:
                    block_content = code_block_match.group(1)
                    try:
                        response_json = json.loads(block_content)
                    except json.JSONDecodeError:
                        try:
                            response_json = ast.literal_eval(block_content)
                        except Exception:
                            pass

                # Method 2: Scan for valid JSON object in the text (Handles \boxed{...} etc)
                if response_json is None:
                    decoder = json.JSONDecoder()
                    idx = 0
                    while True:
                        try:
                            idx = response.find('{', idx)
                            if idx == -1:
                                break
                            obj, end_idx = decoder.raw_decode(response, idx=idx)
                            # Check if it looks like the right object
                            if isinstance(obj, dict) and "action_idx" in obj:
                                response_json = obj
                                break
                            idx += 1 
                        except json.JSONDecodeError:
                            idx += 1
                
                # Method 3: Fallback for Python-dict style (single quotes) embedded in text
                # If we couldn't find valid JSON, try to find a braced-block that works with AST
                if response_json is None:
                    # Find { ... } content - relying on regex is brittle for nesting, but 
                    # we try to grab the largest block or code-block-less content
                    cleaned_response = response.replace("```json", "").replace("```", "").strip()
                    try:
                        response_json = ast.literal_eval(cleaned_response)
                    except Exception:
                        # Try finding just the dict part with regex (last resort)
                        # Non-greedy match for first { to last } is safer than greedy .*, 
                        # but "action_idx" anchor might be better.
                        # Using greedy .* as last resort:
                        try:
                            match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                            if match:
                                response_json = ast.literal_eval(match.group(0))
                        except Exception:
                            pass

                if response_json is None:
                    self.logger.error(f"Failed to parse JSON/AST by all methods: {response}")
                    raise ValueError("Could not parse LLM response")
                
                # Handle multiple possible key names for the selected action
                if "selected_action_index" in response_json:
                    response_json["action_idx"] = response_json["selected_action_index"]
                elif "best_action_idx" in response_json:
                    # New v2 prompt format with top-2 comparison
                    response_json["action_idx"] = response_json["best_action_idx"]
                
                if "action_idx" not in response_json:
                    raise KeyError(f"Response missing 'action_idx', 'selected_action_index', or 'best_action_idx': {response_json}")

                response_json["action_idx_raw"] = response_json["action_idx"]
                response_json["action_idx"] = int(response_json["action_idx"]) - 1
                response_json["raw_llm_response"] = response  # Store raw response for debugging
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(response_json, f, indent=2)
                    self.logger.info(f"Action selection saved to {save_path}")
                
                return response_json
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed for folder {folder}: {e}. Retrying...")
                    self.logger.warning(f"llm_model: {llm_model}, images: {image_paths}")
                    continue
                else:
                    self.logger.error(f"All {max_retries} attempts failed for folder {folder}.")
                    pass
        

def parse_args_and_logger():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="epoch-24", help="Model key to select the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--data_folder", type=str, default="new_prompt/test_mini", help="Folder containing input images and where outputs will be saved.")
    parser.add_argument("--num_options", type=int, default=5, help="Number of action options to generate.")
    parser.add_argument("--mode", type=str, default="text+image", help="Mode for action outcome synthesis: text, image, text+image, none.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for logging.") 
    parser.add_argument("--action_synthesis_llm_model", type=str, default="gpt-4o-20240513", help="LLM model for action outcome synthesis. Supports OpenAI models and Qwen VL models (e.g., Qwen/Qwen3-VL-8B-Instruct).")
    parser.add_argument("--action_option_llm_model", type=str, default="gpt-4o-20240513", help="LLM model for action option generation. Supports OpenAI models and Qwen VL models.")
    parser.add_argument("--action_selection_llm_model", type=str, default="gpt-4o-20240513", help="LLM model for action selection. Supports OpenAI models and Qwen VL models.")
    parser.add_argument("--parallel", type=int, default=-1, help="Whether to run in parallel mode.") 
    parser.add_argument("--ignore_unsuccessful_options", action="store_true", default=False, help="Whether to ignore options that failed during generation.")
    parser.add_argument("--timeout_minutes", type=int, default=5, help="Timeout in minutes for each folder processing. Set to 0 to disable.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for timed-out folders.")
    parser.add_argument("--prompt_version", type=str, default="v1", choices=["v1", "v2", "v3", "default"], help="Prompt version for action selection: v1, v2, v3, or default.")
    parser.add_argument("--qwen_lora_path", type=str, default=None, help="Path to LoRA adapter for Qwen VL models.")
    # gpt-5.2-chat-20251211, gpt-4o-20240513

    args = parser.parse_args()
    curr_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    if args.exp_name is None:
        args.exp_name = f"{curr_time}_n_options_{args.num_options}_mode_{args.mode}_model_{args.model_key}"
    
    # Setup logger
    data_root = os.environ.get("DATA_ROOT", "data")
    log_dir = os.path.join(data_root, "agent_eval_result", args.exp_name, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"agent_eval_{curr_time}.log")
    args_file = os.path.join(log_dir, f"args_{curr_time}.json")
    with open(args_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    
    logger = setup_logger(log_file)
    logger.info(f"Starting experiment: {args.exp_name} at {curr_time}")
    logger.info(f"Arguments: {args}")

    return args, logger
        
        
if __name__ == "__main__":
    args, logger = parse_args_and_logger()
    agent = Agent(args, logger)
    
    folders = []
    for root, dirs, files in os.walk(os.path.join(agent.ROOT, args.data_folder)):
        if "prev.png" in files:
            folders.append(root)
    
    # random.seed(42)
    # # 随机 sample 10 个（假设 folders 数量 ≥ 10）
    # folders = random.sample(folders, 10)
    # print(f"Sampled folders for processing: {folders}")
    
    if args.ignore_unsuccessful_options:
        from eval.action_evaluation import ActionEvaluationA11y
        from eval.agent_wm_eval import get_gt_action, get_all_action_options
        evaluator = ActionEvaluationA11y()
    
            
    def process_one_folder(root):
        logger.info(f"Processing folder: {root}...")
        agent.action_option_generation(
            folder=root, 
            num_options=args.num_options,
            llm_model=args.action_option_llm_model,
        )
        
        if args.ignore_unsuccessful_options:
            eval_folder = agent._get_output_path(root, "")
            with open(agent._get_input_path(root, "action.json"), "r", encoding="utf-8") as f:
                gt_command = json.load(f)
            with open(agent._get_input_path(root, "action_raw.json"), "r", encoding="utf-8") as f:
                gt_raw = json.load(f)
                
            action_options = get_all_action_options(eval_folder)
            
            has_successful_option = False
            
            for idx, option in enumerate(action_options):
                action_command_res = evaluator.compare_action_command(
                    gt_raw=gt_raw,
                    gt_command=gt_command,
                    pred_command=option,
                )["overall_match"]
                
                if action_command_res:
                    has_successful_option = True
                    break
            
            if not has_successful_option:
                logger.info(f"No successful action option found for folder {root}, skipping further processing.")
                return
            
        agent.action_outcome_synthesis(
            mode=args.mode, 
            folder=root, 
            llm_model=args.action_synthesis_llm_model,
        )
        agent.action_selection(
            folder=root,
            llm_model=args.action_selection_llm_model,
        )
        return root
    
    if args.parallel == -1:
        if args.mode in ["image"] and args.model_key != "gpt-image-1.5":
            args.parallel = 1
        else:
            args.parallel = 10

    timeout_seconds = args.timeout_minutes * 60 if args.timeout_minutes > 0 else None
    max_retries = args.max_retries

    def process_with_timeout_and_retry(folder):
        """Process a folder with timeout and retry logic."""
        for attempt in range(max_retries):
            try:
                if timeout_seconds:
                    result = run_with_timeout(
                        process_one_folder, 
                        args=(folder,), 
                        timeout_seconds=timeout_seconds
                    )
                else:
                    result = process_one_folder(folder)
                return result
            except TimeoutError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} timed out for folder {folder}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying folder {folder}...")
                else:
                    logger.error(f"All {max_retries} attempts timed out for folder {folder}, skipping.")
            except Exception as e:
                logger.error(f"Error processing folder {folder}: {e}")
                break  # Don't retry on non-timeout errors
        return None

    if args.parallel > 1:
        num_workers = args.parallel
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(process_with_timeout_and_retry, folders),
                    total=len(folders)))
    else:
        for folder in tqdm(folders):
            process_with_timeout_and_retry(folder)
    
    