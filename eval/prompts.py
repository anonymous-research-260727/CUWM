
"""
This file contains the prompt templates for the GUI agent evaluation.
"""

GUI_ACTION_SELECTION_PROMPT_1 = """
You are an AI assistant tasked with selecting the best next action in an Office application (e.g., Microsoft Word, Excel, or PowerPoint) based on a GUI screenshot and several candidate action outcomes.

The task instruction is:
{request}

The current screenshot is: See the image below.
"""


GUI_ACTION_SELECTION_PROMPT_PREDICTIVE = """
You are an advanced World Model GUI Assistant tasked with selecting the best next action in an Office application (e.g., Microsoft Word, Excel, or PowerPoint).
You will be basing your decision on a GUI screenshot, a set of candidate actions, and **simulated image predictions** generated for each action.

The task instruction is:
{request}

The current screenshot is: See the image below.
"""


# ==========================================
# 1. DEFAULT PROMPT (v1)
# ==========================================

def _build_gui_action_selection_prompt_default(
    include_text: bool,
    include_image: bool,
) -> str:
    fields = [
        "- action: the structured action command to be executed\n",
    ]

    reasoning_note = ""

    # Rename: these are predictions of the future GUI state, not “synthesis” of the action.
    if include_text:
        fields.append(
            "- predicted_state_text: a brief textual prediction of the expected GUI state after executing the action\n"
        )

    if include_image:
        fields.append(
            "- predicted_state_image: a visual prediction (mockup) showing the expected GUI state after executing the action\n"
        )

    if include_text or include_image:
        reasoning_note = "The predicted_state_text and/or predicted_state_image fields are predictions of the GUI state after executing the action. They may not be fully accurate, but you may use them as references when making your decision. If you believe they are unreliable or inconsistent with the screenshot or task, you may ignore them.\n\n"

    return f"""
You are given a list of action options, each with:
{"".join(fields)}

{reasoning_note.rstrip()}

Your goal is to choose the action whose predicted outcome most appropriately advances the user toward their intended goal.

Please return your final answer as a JSON object with the following format:
{{
  "action_idx": <index of selected action>,
  "thought": "<brief explanation of why this action was selected>"
}}

Do NOT include anything else outside the JSON object.

Here are the candidate action options:
""".strip()

GUI_ACTION_SELECTION_NONE_PROMPT_DEFAULT = _build_gui_action_selection_prompt_default(
    include_text=False,
    include_image=False,
)

GUI_ACTION_SELECTION_TEXT_PROMPT_DEFAULT = _build_gui_action_selection_prompt_default(
    include_text=True,
    include_image=False,
)

GUI_ACTION_SELECTION_IMAGE_PROMPT_DEFAULT = _build_gui_action_selection_prompt_default(
    include_text=False,
    include_image=True,
)

GUI_ACTION_SELECTION_TEXT_IMAGE_PROMPT_DEFAULT = _build_gui_action_selection_prompt_default(
    include_text=True,
    include_image=True,
)

GUI_ACTION_SELECTION_PROMPT_DEFAULT = {
    "none": GUI_ACTION_SELECTION_NONE_PROMPT_DEFAULT, 
    "text": GUI_ACTION_SELECTION_TEXT_PROMPT_DEFAULT,
    "image": GUI_ACTION_SELECTION_IMAGE_PROMPT_DEFAULT, 
    "text+image": GUI_ACTION_SELECTION_TEXT_IMAGE_PROMPT_DEFAULT
}


# ==========================================
# 2. IMPROVED PROMPT (v2)
# ==========================================

def _build_gui_action_selection_prompt_improved(
    include_text: bool,
    include_image: bool,
) -> str:
    fields = [
        "- action: the structured action command to be executed\n",
    ]
    reasoning_note = ""
    if include_text:
        fields.append(
            "- predicted_state_text: a brief textual prediction of the expected GUI state after executing the action\n"
        )
    if include_image:
        fields.append(
            "- predicted_state_image: a visual prediction (mockup) showing the expected GUI state after executing the action\n"
        )
    if include_text or include_image:
        # CONCISE NOTE: Explicitly mentions diffusion artifacts
        reasoning_note = (
            "The `predicted_state` fields are simulations. Use them as references, but **ignore diffusion artifacts** "
            "(e.g., garbled text, noisy details). If a prediction is unreliable or inconsistent with the task, "
            "you may ignore it.\n\n"
        )
    return f"""
You are given a list of action options, each with:
{"".join(fields)}
{reasoning_note.rstrip()}

### Analysis Instructions:
1. **Analyze Every Option:** specific `predicted_state` simulations (images/text) are provided for every action. You must look at all of them first.
2. **When to use the World Model:**
   - **Uncertainty:** Use the predicted image when you are uncertain based purely on the action name or your internal knowledge.
   - **Discovery:** Use the predicted image if it clearly shows a better way of advancing the goal that you didn't initially think of.
   - **Efficiency:** Use the predicted image if it shows a way to skip intermediary steps and advance to the goal faster (e.g., a direct edit vs. opening a menu).
3. **Decide:** Combine your internal knowledge with these visual signals to pick the single best action.


Please return your final answer as a JSON object with the following format:
{{
  "action_idx": <index of selected action>,
  "thought": "<brief explanation of why this action was selected>"
}}
Do NOT include anything else outside the JSON object.
Here are the candidate action options:
""".strip()


GUI_ACTION_SELECTION_NONE_PROMPT_IMPROVED = _build_gui_action_selection_prompt_improved(
    include_text=False,
    include_image=False,
)

GUI_ACTION_SELECTION_TEXT_PROMPT_IMPROVED = _build_gui_action_selection_prompt_improved(
    include_text=True,
    include_image=False,
)

GUI_ACTION_SELECTION_IMAGE_PROMPT_IMPROVED = _build_gui_action_selection_prompt_improved(
    include_text=False,
    include_image=True,
)

GUI_ACTION_SELECTION_TEXT_IMAGE_PROMPT_IMPROVED = _build_gui_action_selection_prompt_improved(
    include_text=True,
    include_image=True,
)

GUI_ACTION_SELECTION_PROMPT2 = {
    "none": GUI_ACTION_SELECTION_NONE_PROMPT_IMPROVED, 
    "text": GUI_ACTION_SELECTION_TEXT_PROMPT_IMPROVED,
    "image": GUI_ACTION_SELECTION_IMAGE_PROMPT_IMPROVED, 
    "text+image": GUI_ACTION_SELECTION_TEXT_IMAGE_PROMPT_IMPROVED
}


# ==========================================
# 3. REASONING PROMPT (v2 with strict format)
# ==========================================

def _build_gui_action_selection_reasoning_prompt(
    include_text: bool,
    include_image: bool,
) -> str:
    fields = [
        "- action: the structured action command to be executed\n",
    ]

    reasoning_note = ""

    if include_text:
        fields.append(
            "- predicted_state_text: a brief textual prediction of the expected GUI state after executing the action\n"
        )

    if include_image:
        fields.append(
            "- predicted_state_image: a visual prediction (mockup) showing the expected GUI state after executing the action\n"
        )

    if include_text or include_image:
        reasoning_note = "The predicted_state_text and/or predicted_state_image fields are predictions of the GUI state after executing the action. They may not be fully accurate, but you may use them as references when making your decision. If you believe they are unreliable or inconsistent with the screenshot or task, you may ignore them.\n\n"

    return f"""
You are given a list of action options, each with:
{"".join(fields)}

{reasoning_note.rstrip()}

Your goal is to choose the action whose predicted outcome most appropriately advances the user toward their intended goal.

Please return your final answer as a JSON object with the following format:
{{
  "action_idx": <index of selected action>,
  "thought": "<brief explanation of why this action was selected>",
  "action_selection_reasoning": {{
    "1": "<reasoning for selecting/not selecting option 1>",
    "2": "<reasoning for selecting/not selecting option 2>",
    ...
  }},
  "world_model_feedback": {{
    "helpful": <bool>,
    "usage": "<what did you use the prediction for>",
    "explanation": "<how did it help>"
  }}
}}

The "action_selection_reasoning" field should contain a dictionary where keys are the option indices (as strings usually, e.g. "1", "2", ...) and values are your reasoning for why you selected or didn't select that specific option.
The "world_model_feedback" field should indicate if the world model predictions (text or image) were helpful in your decision making process.

Do NOT include anything else outside the JSON object.

Here are the candidate action options:
""".strip()


GUI_ACTION_SELECTION_NONE_REASONING_PROMPT = _build_gui_action_selection_reasoning_prompt(
    include_text=False,
    include_image=False,
)

GUI_ACTION_SELECTION_TEXT_REASONING_PROMPT = _build_gui_action_selection_reasoning_prompt(
    include_text=True,
    include_image=False,
)

GUI_ACTION_SELECTION_IMAGE_REASONING_PROMPT = _build_gui_action_selection_reasoning_prompt(
    include_text=False,
    include_image=True,
)

GUI_ACTION_SELECTION_TEXT_IMAGE_REASONING_PROMPT = _build_gui_action_selection_reasoning_prompt(
    include_text=True,
    include_image=True,
)

GUI_ACTION_SELECTION_REASONING_PROMPT2 = {
    "none": GUI_ACTION_SELECTION_NONE_REASONING_PROMPT, 
    "text": GUI_ACTION_SELECTION_TEXT_REASONING_PROMPT,
    "image": GUI_ACTION_SELECTION_IMAGE_REASONING_PROMPT, 
    "text+image": GUI_ACTION_SELECTION_TEXT_IMAGE_REASONING_PROMPT
}


# ==========================================
# 4. PREDICTIVE PROMPT (v3)
# ==========================================

def _build_gui_action_selection_prompt_predictive(
    include_text: bool,
    include_image: bool,
) -> str:
    
    # New task overview focusing on predictive causality
    return """
You are given a list of action options. For each option, you will see the proposed action and a prediction of what will happen if you execute it.

**CRITICAL INSTRUCTION:** You represent a high-stakes agent. You **MUST** examine the prediction (image/text) for **EVERY** candidate action provided below. Compare the simulated outcomes of all options against the user's request before making a selection. Do not rely solely on the action names; verify the actual visual outcome in the world model.

The text and/or image predictions represent simulations from a World Model. You should primarily rely on the current screenshot and the `action` definition. However, you should consult these predictions in two specific scenarios:
1. **Uncertainty:** If you are unsure about the outcome of the primary action.
2. **Alternative Discovery:** Look into the world model outputs to help consider alternative actions you may have yet to consider, preventing you from overlooking better options.

Since these are simulated predictions, they may contain minor artifacts. You should:
1. **Rely on the predictions** to judge the outcome, unless you observe **obvious and severe hallucinations** that completely contradict the action's intent. Do not discard predictions for minor issues.
2. **Ignore minor visual noise** in predicted images, such as garbled text, slightly imperfect rendering, or small details. Focus on the overall structural changes and whether the state moves towards the goal.

Ask yourself: "If I take this action, does the resulting image/text actually match the user's goal?"

Your goal is to choose the action whose predicted outcome most appropriately advances the user toward their intended goal.

Please return your final answer as a JSON object with the following format:
{{
  "action_idx": <index of selected action>,
  "thought": "<brief explanation of why this action was selected>"
}}

Do NOT include anything else outside the JSON object.

Here are the candidate action options:
""".strip()


GUI_ACTION_SELECTION_NONE_PROMPT_PREDICTIVE = _build_gui_action_selection_prompt_predictive(
    include_text=False,
    include_image=False,
)

GUI_ACTION_SELECTION_TEXT_PROMPT_PREDICTIVE = _build_gui_action_selection_prompt_predictive(
    include_text=True,
    include_image=False,
)

GUI_ACTION_SELECTION_IMAGE_PROMPT_PREDICTIVE = _build_gui_action_selection_prompt_predictive(
    include_text=False,
    include_image=True,
)

GUI_ACTION_SELECTION_TEXT_IMAGE_PROMPT_PREDICTIVE = _build_gui_action_selection_prompt_predictive(
    include_text=True,
    include_image=True,
)


GUI_ACTION_SELECTION_PROMPT3 = {
    "none": GUI_ACTION_SELECTION_NONE_PROMPT_PREDICTIVE, 
    "text": GUI_ACTION_SELECTION_TEXT_PROMPT_PREDICTIVE,
    "image": GUI_ACTION_SELECTION_IMAGE_PROMPT_PREDICTIVE, 
    "text+image": GUI_ACTION_SELECTION_TEXT_IMAGE_PROMPT_PREDICTIVE
}


# ==========================================
# 5. SUPPORTED ACTIONS & SCORING
# ==========================================

SUPPORTED_ACTIONS_WORD = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- insert_table
  - Args:
    - rows: int, The number of rows in the table, starting from 1.
    - columns: int, The number of columns in the table, starting from 1.
  - Example: insert_table(rows=3, columns=3)
- select_text
  - Args:
    - text: str, The exact text to be selected.
  - Example: select_text(text="Hello")
- select_table
  - Args:
    - number: int, The index number of the table to be selected.
  - Example: select_table(number=1)
- select_paragraph
  - Args:
    - start_index: int, The start index of the paragraph to be selected.
    - end_index: int, The end index of the paragraph, if ==-1, select to the end of the document.
    - non_empty: bool, If True, select the non-empty paragraphs only. (Default: True)
  - Example: select_paragraph(start_index=1, end_index=3, non_empty=True)
- save_as
  - Args:
    - file_dir: str, The directory to save the file. If not specified, the current directory will be used. (Default: "")
    - file_name: str, The name of the file without extension. If not specified, the name of the current document will be used. (Default: "")
    - file_ext: str, The extension of the file. If not specified, the default extension is ".pdf". (Default: ".pdf")
  - Example: save_as(file_dir="", file_name="", file_ext=".pdf")
- set_font
  - Args:
    - font_name: str|None, The name of the font (e.g., "Arial", "Times New Roman", "宋体"). If None, the font name will not be changed. (Default: None)
    - font_size: int|None, The font size (e.g., 12). If None, the font size will not be changed. (Default: None)
  - Example: set_font(font_name="Times New Roman")
</action>"""

SUPPORTED_ACTIONS_EXCEL = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- table2markdown
  - Args:
    - sheet_name: str|int, The name or index of the sheet to get the table content. The index starts from 1.
  - Example: table2markdown(sheet_name=1)
- insert_excel_table
  - Args:
    - table: list[list], The table content to insert. The table is a list of list of strings or numbers.
    - sheet_name: str, The name of the sheet to insert the table.
    - start_row: int, The start row to insert the table, starting from 1.
    - start_col: int, The start column to insert the table, starting from 1.
  - Example: insert_excel_table(table=[["Name", "Age", "Gender"], ["Alice", 30, "Female"], ["Bob", 25, "Male"], ["Charlie", 35, "Male"]], sheet_name="Sheet1", start_row=1, start_col=1)
- select_table_range
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The start row, starting from 1.
    - start_col: int, The start column, starting from 1.
    - end_row: int, The end row. If ==-1, select to the end of the document with content.
    - end_col: int, The end column. If ==-1, select to the end of the document with content.
  - Example: select_table_range(sheet_name="Sheet1", start_row=1, start_col=1, end_row=3, end_col=3)
- set_cell_value
  - Args:
    - sheet_name: str, The name of the sheet.
    - row: int, The row number (1-based).
    - col: int, The column number (1-based).
    - value: str|int|float|None, The value to set in the cell. If None, just select the cell.
    - is_formula: bool, If True, treat the value as a formula, otherwise treat it as a normal value. (Default: False)
  - Example: set_cell_value(sheet_name="Sheet1", row=1, col=1, value="Hello", is_formula=False), set_cell_value(sheet_name="Sheet1", row=2, col=2, value="=SUM(A1:A10)", is_formula=True)
- auto_fill
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The starting row number (1-based).
    - start_col: int, The starting column number (1-based).
    - end_row: int, The ending row number (1-based).
    - end_col: int, The ending column number (1-based).
  - Example: auto_fill(sheet_name="Sheet1", start_row=1, start_col=1, end_row=10, end_col=3)
- reorder_columns
  - Args:
    - sheet_name: str, The name of the sheet.
    - desired_order: list[str], The list of column names in the new order.
  - Example: reorder_columns(sheet_name="Sheet1", desired_order=["Income", "Date", "Expense"]) 
</action>"""

SUPPORTED_ACTIONS_PPT = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- set_background_color
  - Args:
    - color: str, The hex color code (in RGB format) to set the background color.
    - slide_index: list[int]|None, The list of slide indexes to set the background color. If None, set the background color for all slides.
  - Example: set_background_color(color="FFFFFF", slide_index=[1, 2, 3])
- save_as
  - Args:
    - file_dir: str, The directory to save the file. If not specified, the current directory will be used. (Default: "")
    - file_name: str, The name of the file without extension. If not specified, the name of the current document will be used. (Default: "")
    - file_ext: str, The extension of the file. If not specified, the default extension is ".pptx". (Default: ".pptx")
    - current_slide_only: bool, This only applies to ".jpg", ".png", ".gif", ".bmp" and ".tiff" formats. If True, only the current slide will be saved to a PNG file. If False, all slides will be saved into a directory containing multiple PNG files. (Default: False)
  - Example: save_as(file_dir="", file_name="", file_ext=".pdf")
</action>"""

SUPPORTED_ACTIONS = {
    "word": SUPPORTED_ACTIONS_WORD,
    "excel": SUPPORTED_ACTIONS_EXCEL,
    "ppt": SUPPORTED_ACTIONS_PPT,
}


ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT = """
You are an expert in Office Application automation and graphical user interfaces with accessibility support.

You will be provided with the following inputs:

1. **Current screenshot**: An image of the current state of an Office Application.
2. **Annotated current screenshot**: The same screenshot annotated with numeric markers corresponding to accessibility elements.2. **Annotated current screenshot**: The same screenshot annotated with numeric markers corresponding to accessibility elements.
3. **Accessibility (a11y) information**: This includes a list of control element labels and the textual name of the currently active Office Application.
4. **Task instruction**: A description of the action or goal to be completed.
5. **Supported actions**: A list of all actions that can be performed in this environment.

The annotated screenshot contains numbers that correspond directly to entries in the accessibility information. Each number identifies a specific UI control element, allowing you to reliably locate and reference interface components.

The accessibility information contains control labels that correspond to UI control elements in the current screenshot, allowing you to locate and reference specific interface components.

Your objective is to generate {num_options} COMPLETELY UNIQUE and diverse next actions that could be taken to accomplish the given task instruction, based on the current screenshot of the Office Application, the annotated current screenshot, and the available accessibility information. The desired number of options will be specified in the user instruction or input.

Since you are powering a World Model, we need to explore different branches of possibilities. You must strictly avoid duplicates.

Use all the provided information—including the current screenshot, annotated screenshot, accessibility data, task instruction, and supported actions—to reason about the next appropriate actions accurately.

**IMPORTANT: When possible, prioritize using control_label over coordinate for actions. Control labels refer to unique identifiers provided by the accessibility (a11y) system for UI elements (such as buttons, text fields, or menu items). These identifiers are more reliable and accessible than raw screen coordinates, which may vary across layouts, resolutions, or UI states.**

For each candidate action, explain your reasoning process—describe how you analyze the current screenshot and the annotated screenshot, interpret the accessibility information, understand the current UI state, and determine what action could be taken next to move toward completing the task instruction.

Then, output the next actions in JSON format as a JSON array. Each element of the array must be an object with the keys "thoughts", "uniqueness_check", and "tool_call". All three fields MUST be present in every element.

Please think very hard and carefully about the current state and the task instruction before making a decision, and output your reasoning in each element's "thoughts" field as detailed as possible, including:
- Your analysis of the current screenshot and annotated screenshot
- Your interpretation of the accessibility information
- How you identified the target control using control labels or visual elements
- The reason for your tool call and argument selection
- Your assessment of task progress based on the current state

The "uniqueness_check" field must strictly verify that this option is distinct from all previous options in the list. You must explicitly state: "Action [X] is distinct from previous actions because it targets [Different Element]..."

The "tool_call" field in each element should contain:
- "function": str, The function/action type to execute
- "args": Dict, The arguments/parameters for the function
- "status": str, The status after performing this action (either "CONTINUE" or "FINISH")

For click operations, prioritize control_label over coordinate:
```json
{
  "thoughts": "The screenshot shows an Excel spreadsheet. From the accessibility information, there is a Save button with control_label=15. Using the control_label provides a more reliable interaction than estimating screen coordinates.",
  "uniqueness_check": "This is Option 1, so it is unique by default.",
  "tool_call": {
    "function": "click",
    "args": {"control_label": 15, "coordinate": null, "button": "left"},
    "status": "CONTINUE"
  }
}
````

If control_label is not available, fall back to coordinate:

```json
{
  "thoughts": "The target UI element does not have a corresponding control_label in the accessibility information. Based on the visual analysis of the annotated screenshot, I estimate the appropriate screen coordinates to interact with it.",
  "uniqueness_check": "Option 2 (Click Coordinate) is distinct from Option 1 (Click Label) because it targets a different UI element at specific coordinates.",
  "tool_call": {
    "function": "click",
    "args": {"control_label": null, "coordinate": [150, 30], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For type operations, prioritize control_label over coordinate:

```json
{
  "thoughts": "The accessibility information indicates a text input field with control_label=8. Typing via the control_label ensures the correct field is targeted.",
  "uniqueness_check": "Option 3 (Type) is distinct from previous Click actions.",
  "tool_call": {
    "function": "type",
    "args": {"control_label": 8, "coordinate": null, "keys": "Hello World", "clear_current_text": true},
    "status": "CONTINUE"
  }
}
```

For drag operations, coordinates are required:

```json
{
  "thoughts": "This task requires dragging an element from one location to another. Drag actions require explicit start and end coordinates to describe the spatial movement.",
  "uniqueness_check": "Option 4 (Drag) is distinct from all previous actions.",
  "tool_call": {
    "function": "drag",
    "args": {"start_coordinate": [100, 100], "end_coordinate": [200, 200], "button": "left"},
    "status": "CONTINUE"
  }
}
```

If you think the task is finished, output status as "FINISH":

```json
{
  "thoughts": "Based on the current state of the Office Application and the task instruction, no further actions are required.",
  "uniqueness_check": "Finish action.",
  "tool_call": {
    "function": "",
    "args": {},
    "status": "FINISH"
  }
}
```

Only **ONE** action should be taken per option. You may output multiple options (as requested by num_options), but each option must correspond to exactly one action. If the task instruction could apply to multiple elements, choose the most relevant one for each option based on the current screenshot and accessibility information, and ensure the options are diverse.

**STRICT CONSTRAINT ON DUPLICATES:**
- **Semantic Uniqueness:** You MUST NOT target the same UI element twice using different methods.
- **Forbidden:** Generating one option that clicks "Save" via `control_label` and another that clicks "Save" via `coordinate`. This is a duplicate.
- **Verification:** You MUST use the `uniqueness_check` field to explicitly validate difference from ALL previous options.

Your response MUST be a valid JSON array with no additional text outside the JSON structure.
"""

ACTION_PREDICTION_A11Y_USER_PROMPT_GPT = """
Task instruction:
{instruction}

Accessibility Information:
{a11y}

Supported actions:
{actions}

The current screenshot and the annotated current screenshot are provided as images.

Please analyze the current state using both visual information and accessibility data to generate {num_options} UNIQUE and plausible next actions that could be taken to move toward completing the task instruction.

**REMINDER: Each action option MUST be semantically unique. Do NOT target the same element twice. Verify in your `uniqueness_check` field that this action interacts with a completely different part of the screen than the previous options.**

Please provide your reasoning, uniqueness check, and the next actions below in JSON array format without any additional text.
"""

GUI_BATCH_VALIDATION_PROMPT_TEMPLATE = """
You are an AI assistant tasked with evaluating multiple potential UI actions on an Android device based on the user's goal, the current state, and predicted future states.

The task instruction is:
{request}

The current screenshot is: See the image below.
"""



