# TOOD-yr: 这个 prompt 要和 eval.prompts 里的保持一致
import json
from utils.cloudgpt_aoai import get_chat_completion, encode_image
from eval.action_evaluation import ActionEvaluationA11y
from eval.prompts import SUPPORTED_ACTIONS

actEval = ActionEvaluationA11y()


ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT = """
You are an expert in Office Application automation and graphical user interfaces with accessibility support.

You will be provided with the following inputs:

1. **Current screenshot (optional)**: An image of the current state of an Office Application. This screenshot may be missing.
2. **Screenshot Description**: A textual description of the current UI state derived from the screenshot.
3. **Accessibility (a11y) information**: This includes a list of control element labels and the textual name of the currently active Office Application.
4. **Task instruction**: A description of the action or goal to be completed.
5. **Supported actions**: A list of all actions that can be performed in this environment.

The accessibility information contains control labels that correspond to UI control elements in the current application state, allowing you to locate and reference specific interface components.

Your objective is to generate the **single best next action** to accomplish the given task instruction, based on the available information, including the screenshot description, accessibility information, task instruction, supported actions, and the current screenshot if it is provided.

Use all the provided information to determine the most appropriate next action. If the current screenshot is not available, rely on the screenshot description and accessibility information.

**IMPORTANT: When possible, prioritize using control_label over coordinate for actions. Control labels are more reliable than raw screen coordinates.**

You must output the next action in JSON format as a JSON array containing **exactly one element**.

Each element must contain only a "tool_call" field.

The "tool_call" field must contain:
- "function": str, The function/action type to execute
- "args": Dict, The arguments/parameters for the function
- "status": str, The status after performing this action (either "CONTINUE" or "FINISH")

Only **ONE** action should be generated.

For example, for click operations, prioritize control_label over coordinate:
```json
{
  "tool_call": {
    "function": "click",
    "args": {"control_label": 15, "coordinate": null, "button": "left"},
    "status": "CONTINUE"
  }
}
````

For example, if control_label is not available, fall back to coordinate:

```json
{
  "tool_call": {
    "function": "click",
    "args": {"control_label": null, "coordinate": [150, 30], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For example, for type operations:

```json
{
  "tool_call": {
    "function": "type",
    "args": {"control_label": 8, "coordinate": null, "keys": "Hello World", "clear_current_text": true},
    "status": "CONTINUE"
  }
}
```

For example, for drag operations:

```json
{
  "tool_call": {
    "function": "drag",
    "args": {"start_coordinate": [100, 100], "end_coordinate": [200, 200], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For example, if the task is already completed, output:

```json
{
  "tool_call": {
    "function": "",
    "args": {},
    "status": "FINISH"
  }
}
```

Your response MUST be a valid JSON array with exactly one element and no additional text.
"""

ACTION_PREDICTION_A11Y_USER_PROMPT_GPT = """
Task instruction:
{instruction}

Screenshot Description:
{screen_description}

Accessibility Information:
{a11y}

Supported actions:
{actions}

The current screenshot may be provided as an image, but it may also be missing.

Please analyze the current state using the available information and output the **single best next action** to move toward completing the task instruction.

Output the result in JSON array format (with exactly one element) and no additional text.
"""


def generate_prompt(
    screen_description: str | None,
    screen_description_path: str | None,
    a11y_path: str,
    instruction_path: str,
):
    assert screen_description or screen_description_path
    assert not (screen_description and screen_description_path)
    if screen_description_path:
        with open(screen_description_path, "r", encoding="utf-8") as f:
            screen_description = f.read().strip()
    
    with open(a11y_path, "r", encoding="utf-8") as f:
        a11y = json.load(f)
    
    with open(instruction_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()
        
    if "ppt" in a11y_path:
       app = "ppt"
    elif "word" in a11y_path:
        app = "word"
    elif "excel" in a11y_path:
        app = "excel"
    else:
        raise ValueError(f"Cannot infer app type from folder: {a11y_path}")
    
    supported_actions = SUPPORTED_ACTIONS[app]
    
    
    usr_prompt = ACTION_PREDICTION_A11Y_USER_PROMPT_GPT.format(
        instruction=instruction,
        screen_description=screen_description,
        a11y=json.dumps(a11y, indent=2),
        actions=supported_actions,
    )
    
    prompt = ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT + "\n\n" + usr_prompt
    return prompt
    
def get_action(
    img_path: str | None, # should be pair_0{i+1}'s prev.png
    screen_description: str | None,
    screen_description_path: str | None,  # should be pair_0{i+1}'s gt_xxx.txt
    a11y_path: str,  # should be pair_0{i+1}'s a11y.json
    instruction_path: str,  # should be pair_0{i+1}'s request.txt
):
    prompt = generate_prompt(
        screen_description=screen_description,
        screen_description_path=screen_description_path,
        a11y_path=a11y_path,
        instruction_path=instruction_path,
    )
    
    content = [
        {"type": "text", "text": prompt}, 
        {"type": "image_url", "image_url": {"url": encode_image(img_path)}},
    ]
    messages = [{"role": "user", "content": content}]
    
    for attempt in range(3):
        try:
            response = get_chat_completion(
                model="qwen8b??",
                messages=messages,
                temperature=0,
            )

            text = response.choices[0].message.content
            text = text.replace("```json", "").replace("```", "").strip()
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            print(f"[Attempt {attempt}] exception:", e)
        
    return None
    
    

def action_coherence_reward(
    textual_wm_response: str,
    img_path: str, # should be pair_0{i+1}'s prev.png
    gt_description_path: str,  # should be pair_0{i+1}'s gt_xxx.txt
    a11y_path: str,  # should be pair_0{i+1}'s a11y.json
    instruction_path: str,  # should be pair_0{i+1}'s request.txt
):
    action_pred = get_action(
        img_path=None,
        screen_description=textual_wm_response,
        a11y_path=a11y_path,
        instruction_path=instruction_path,
    )
    action_gt = get_action(
        img_path=img_path,
        screen_description_path=gt_description_path,
        a11y_path=a11y_path,
        instruction_path=instruction_path,
    )
    
    if action_pred is None or action_gt is None:
        return 0
    
    eval_result = actEval.compare_action_command(
        gt_raw=None,
        gt_command=action_gt,
        pred_command=action_pred,
    )
    function_match = eval_result["function_match"]
    status_match = eval_result["status_match"]
    args_match = eval_result["args_match"]
    
    score = 0
    
    if function_match:
        score += 0.25
    if status_match:
        score += 0.25
    if args_match:
        score += 0.5
    
    return score