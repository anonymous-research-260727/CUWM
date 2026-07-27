GUI_NEXT_SCREEN_WITH_CONTROL_PROMPT = """
The next UI screenshot after the following action is performed:

Action:
{action}

Control Info:
{control_info}

Show the resulting screen with the updated UI after this action.
""".strip()

GUI_NEXT_SCREEN_WO_CONTROL_PROMPT = """
The next UI screenshot after the following action is performed:

Action:
{action}

Show the resulting screen with the updated UI after this action.
""".strip()

GUI_NEXT_SCREEN_WITH_CONTROL_WITH_DESCRIPTION_PROMPT = """
The next UI screenshot after the following action is performed:

Action:
{action}

Control Info:
{control_info}

GUI Action Description:
{gui_description}

Show the resulting screen with the updated UI after this action.
""".strip()

GUI_NEXT_SCREEN_WO_CONTROL_WITH_DESCRIPTION_PROMPT = """
The next UI screenshot after the following action is performed:

Action:
{action}

GUI Action Description:
{gui_description}

Show the resulting screen with the updated UI after this action.
""".strip()



WORD_CLICK_DESCRIPTION = """
- click
  - Args:
      - control_label: int | None
        Description: The label of the control to click. You should prioritize using `control_label` whenever possible. Only use `None` if the target location does not have a corresponding control label. The label must exist in the provided accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to click, used only when the target is not listed in the a11y information and has no control label. In this case, set `control_label` to `None` and estimate the coordinate.
      - button: str
        Description: The mouse button to click. Options are `'left'`, `'right'`, `'middle'`, or `'x'`. Default is `'left'`.
      - double: bool
        Description: Whether to perform a double click. Default is `False`.
      - pressed: str | None
        Description: The keyboard key to hold while clicking. Common keys include `'CONTROL'` (Ctrl), `'SHIFT'` (Shift), `'MENU'` (Alt), etc. Do not include prefixes like `VK_` or braces. For example, use `'CONTROL'` for the Control key. Default is `None`.
""".strip()

WORD_TYPE_DESCRIPTION = """
- type
  - Args:
      - control_label: int | None
        Description: The label of the control to type into. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to type at, used only when the target control is not listed in the a11y information. Set `control_label` to `None` and use this coordinate.
      - keys: str
        Description: The keys to type. Can include any keyboard key, with special keys represented by their virtual key codes (e.g., `"{VK_CONTROL}c"` for Ctrl+C).
      - clear_current_text: bool
        Description: Whether to clear the current text in the control before typing new text. Default is `False`.
      - control_focus: bool
        Description: Whether to focus the selected control before typing. If `False`, hotkeys will be sent to the application window. Default is `True`.
""".strip()

WORD_DRAG_DESCRIPTION = """
- drag
  - Args:
      - start_coordinate: [x, y]
        Description: The absolute screen position where the drag starts.
      - end_coordinate: [x, y]
        Description: The absolute screen position where the drag ends.
      - button: str
        Description: The mouse button to use for dragging. Options: `'left'` or `'right'`. Default is `'left'`.
      - duration: float
        Description: Duration of the drag action in seconds. Default is `1.0`.
      - key_hold: str | None
        Description: Keyboard key to hold while dragging. Common keys include `'shift'`, `'control'`, `'alt'`, etc. Use lowercase names. Default is `None`.
""".strip()

WORD_WHEEL_MOUSE_INPUT_DESCRIPTION = """
- wheel_mouse_input
  - Args:
      - control_label: int | None
        Description: The label of the control for wheel input. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the a11y information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to scroll, used only when the target control is not in the a11y information. Set `control_label` to `None` and use this coordinate.
      - wheel_dist: int
        Description: Number of wheel notches to scroll. Positive values indicate upward scrolling; negative values indicate downward scrolling.
""".strip()

WORD_INSERT_TABLE_DESCRIPTION = """
- insert_table
  - Args:
      - rows: int
        Description: The number of rows in the table.
      - columns: int
        Description: The number of columns in the table.
""".strip()

WORD_SELECT_TEXT_DESCRIPTION = """
- select_text
  - Args:
      - text: str
        Description: The exact text to be selected in the document.
""".strip()

WORD_SELECT_TABLE_DESCRIPTION = """
- select_table
  - Args:
      - number: int
        Description: The index number of the table to be selected (1-based indexing).
""".strip()

WORD_SELECT_PARAGRAPH_DESCRIPTION = """
- select_paragraph
  - Args:
      - start_index: int
        Description: The start index of the paragraph to select (1-based indexing).
      - end_index: int
        Description: The end index of the paragraph. If set to -1, select until the end of the document.
      - non_empty: bool
        Description: If True, only select non-empty paragraphs. Default is True.
""".strip()

WORD_SAVE_AS_DESCRIPTION = """
- save_as
  - Args:
      - file_dir: str
        Description: Directory to save the file. Defaults to the current directory if not specified. Default is `""`.
      - file_name: str
        Description: Name of the file without extension. Defaults to the current document's name if not specified. Default is `""`.
      - file_ext: str
        Description: File extension. Defaults to `.pdf` if not specified. Default is `.pdf`.
""".strip()

WORD_SET_FONT_DESCRIPTION = """
- set_font
  - Args:
      - font_name: str | None
        Description: The name of the font (e.g., `"Arial"`, `"Times New Roman"`, `"宋体"`). If `None`, the font name will not be changed. Default is `None`.
      - font_size: int | None
        Description: The font size (e.g., `12`). If `None`, the font size will not be changed. Default is `None`.
""".strip()

EXCEL_CLICK_DESCRIPTION = """
- click
  - Args:
      - control_label: int | None
        Description: The label of the control to click. You should prioritize using `control_label` whenever possible. Only use `None` if the target location does not have a corresponding control label. The label must exist in the provided accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to click, used only when the target is not listed in the a11y information and has no control label. In this case, set `control_label` to `None` and estimate the coordinate.
      - button: str
        Description: The mouse button to click. Options are `'left'`, `'right'`, `'middle'`, or `'x'`. Default is `'left'`.
      - double: bool
        Description: Whether to perform a double click. Default is `False`.
      - pressed: str | None
        Description: The keyboard key to hold while clicking. Common keys include `'CONTROL'` (Ctrl), `'SHIFT'` (Shift), `'MENU'` (Alt), etc. Do not include prefixes like `VK_` or braces. For example, use `'CONTROL'` for the Control key. Default is `None`.
""".strip()

EXCEL_TYPE_DESCRIPTION = """
- type
  - Args:
      - control_label: int | None
        Description: The label of the control to type into. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to type at, used only when the target control is not listed in the a11y information. Set `control_label` to `None` and use this coordinate.
      - keys: str
        Description: The keys to type. Can include any keyboard key, with special keys represented by their virtual key codes (e.g., `"{VK_CONTROL}c"` for Ctrl+C).
      - clear_current_text: bool
        Description: Whether to clear the current text in the control before typing new text. Default is `False`.
      - control_focus: bool
        Description: Whether to focus the selected control before typing. If `False`, hotkeys will be sent to the application window. Default is `True`.
""".strip()

EXCEL_DRAG_DESCRIPTION = """
- drag
  - Args:
      - start_coordinate: [x, y]
        Description: The absolute screen position where the drag starts.
      - end_coordinate: [x, y]
        Description: The absolute screen position where the drag ends.
      - button: str
        Description: The mouse button to use for dragging. Options: `'left'` or `'right'`. Default is `'left'`.
      - duration: float
        Description: Duration of the drag action in seconds. Default is `1.0`.
      - key_hold: str | None
        Description: Keyboard key to hold while dragging. Common keys include `'shift'`, `'control'`, `'alt'`, etc. Use lowercase names. Default is `None`.
""".strip()

EXCEL_WHEEL_MOUSE_INPUT_DESCRIPTION = """
- wheel_mouse_input
  - Args:
      - control_label: int | None
        Description: The label of the control for wheel input. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the a11y information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to scroll, used only when the target control is not in the a11y information. Set `control_label` to `None` and use this coordinate.
      - wheel_dist: int
        Description: Number of wheel notches to scroll. Positive values indicate upward scrolling; negative values indicate downward scrolling.
""".strip()

EXCEL_TABLE2MARKDOWN_DESCRIPTION = """
- table2markdown
  - Args:
      - sheet_name: str | int
        Description: The name or 1-based index of the sheet from which to extract the table content.
""".strip()

EXCEL_INSERT_EXCEL_TABLE_DESCRIPTION = """
- insert_excel_table
  - Args:
      - table: list[list[str | int | float]]
        Description: The table content to insert. Represented as a list of rows, each row being a list of strings or numbers.
      - sheet_name: str
        Description: The name of the sheet to insert the table into.
      - start_row: int
        Description: The starting row to insert the table, 1-based indexing.
      - start_col: int
        Description: The starting column to insert the table, 1-based indexing.
""".strip()

EXCEL_SELECT_TABLE_RANGE_DESCRIPTION = """
- select_table_range
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - start_row: int
        Description: The start row for selection, 1-based indexing.
      - start_col: int
        Description: The start column for selection, 1-based indexing.
      - end_row: int
        Description: The end row for selection. If set to -1, select until the last row with content.
      - end_col: int
        Description: The end column for selection. If set to -1, select until the last column with content.
""".strip()

EXCEL_SET_CELL_VALUE_DESCRIPTION = """
- set_cell_value
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - row: int
        Description: The row number of the cell, 1-based indexing.
      - col: int
        Description: The column number of the cell, 1-based indexing.
      - value: str | int | float | None
        Description: The value to set in the cell. If `None`, only select the cell without changing its content.
      - is_formula: bool
        Description: If True, treat `value` as a formula; otherwise treat it as a normal value. Default is `False`.
""".strip()

EXCEL_AUTO_FILL_DESCRIPTION = """
- auto_fill
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - start_row: int
        Description: The starting row of the auto-fill range, 1-based indexing.
      - start_col: int
        Description: The starting column of the auto-fill range, 1-based indexing.
      - end_row: int
        Description: The ending row of the auto-fill range, 1-based indexing.
      - end_col: int
        Description: The ending column of the auto-fill range, 1-based indexing.
""".strip()

EXCEL_REORDER_COLUMNS_DESCRIPTION = """
- reorder_columns
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - desired_order: list[str]
        Description: A list of column names specifying the new order.
""".strip()

PPT_CLICK_DESCRIPTION = """
- click
  - Args:
      - control_label: int | None
        Description: The label of the control to click. You should prioritize using `control_label` whenever possible. Only use `None` if the target location does not have a corresponding control label. The label must exist in the provided accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to click, used only when the target is not listed in the a11y information and has no control label. In this case, set `control_label` to `None` and estimate the coordinate.
      - button: str
        Description: The mouse button to click. Options are `'left'`, `'right'`, `'middle'`, or `'x'`. Default is `'left'`.
      - double: bool
        Description: Whether to perform a double click. Default is `False`.
      - pressed: str | None
        Description: The keyboard key to hold while clicking. Common keys include `'CONTROL'` (Ctrl), `'SHIFT'` (Shift), `'MENU'` (Alt), etc. Do not include prefixes like `VK_` or braces. For example, use `'CONTROL'` for the Control key. Default is `None`.
""".strip()

PPT_TYPE_DESCRIPTION = """
- type
  - Args:
      - control_label: int | None
        Description: The label of the control to type into. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to type at, used only when the target control is not listed in the a11y information. Set `control_label` to `None` and use this coordinate.
      - keys: str
        Description: The keys to type. Can include any keyboard key, with special keys represented by their virtual key codes (e.g., `"{VK_CONTROL}c"` for Ctrl+C).
      - clear_current_text: bool
        Description: Whether to clear the current text in the control before typing new text. Default is `False`.
      - control_focus: bool
        Description: Whether to focus the selected control before typing. If `False`, hotkeys will be sent to the application window. Default is `True`.
""".strip()

PPT_DRAG_DESCRIPTION = """
- drag
  - Args:
      - start_coordinate: [x, y]
        Description: The absolute screen position where the drag starts.
      - end_coordinate: [x, y]
        Description: The absolute screen position where the drag ends.
      - button: str
        Description: The mouse button to use for dragging. Options: `'left'` or `'right'`. Default is `'left'`.
      - duration: float
        Description: Duration of the drag action in seconds. Default is `1.0`.
      - key_hold: str | None
        Description: Keyboard key to hold while dragging. Common keys include `'shift'`, `'control'`, `'alt'`, etc. Use lowercase names. Default is `None`.
""".strip()

PPT_WHEEL_MOUSE_INPUT_DESCRIPTION = """
- wheel_mouse_input
  - Args:
      - control_label: int | None
        Description: The label of the control for wheel input. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the a11y information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to scroll, used only when the target control is not in the a11y information. Set `control_label` to `None` and use this coordinate.
      - wheel_dist: int
        Description: Number of wheel notches to scroll. Positive values indicate upward scrolling; negative values indicate downward scrolling.
""".strip()

PPT_SET_BACKGROUND_COLOR_DESCRIPTION = """
- set_background_color
  - Args:
      - color: str
        Description: The background color to set, specified as a hex RGB code (e.g., "FFFFFF" for white).
      - slide_index: list[int] | None
        Description: A list of slide indexes to apply the background color to. If `None`, the color will be applied to all slides.
""".strip()

PPT_SAVE_AS_DESCRIPTION = """
- save_as
  - Args:
      - file_dir: str
        Description: Directory to save the file. Defaults to the current directory if not specified. Default is `""`.
      - file_name: str
        Description: Name of the file without extension. Defaults to the current presentation's name if not specified. Default is `""`.
      - file_ext: str
        Description: File extension. Defaults to `.pptx` if not specified. Default is `.pptx`.
      - current_slide_only: bool
        Description: Applies only when saving as `.jpg`, `.png`, `.gif`, `.bmp`, or `.tiff`. If `True`, only the current slide will be saved; if `False`, all slides will be saved as separate files in a directory. Default is `False`.
""".strip()

GUI_DESCRIPTION = {
    "word": {
        "click": WORD_CLICK_DESCRIPTION,
        "type": WORD_TYPE_DESCRIPTION,
        "drag": WORD_DRAG_DESCRIPTION,
        "wheel_mouse_input": WORD_WHEEL_MOUSE_INPUT_DESCRIPTION,
        "insert_table": WORD_INSERT_TABLE_DESCRIPTION,
        "select_text": WORD_SELECT_TEXT_DESCRIPTION,
        "select_table": WORD_SELECT_TABLE_DESCRIPTION,
        "select_paragraph": WORD_SELECT_PARAGRAPH_DESCRIPTION,
        "save_as": WORD_SAVE_AS_DESCRIPTION,
        "set_font": WORD_SET_FONT_DESCRIPTION,
    },
    "excel": {
        "click": EXCEL_CLICK_DESCRIPTION,
        "type": EXCEL_TYPE_DESCRIPTION,
        "drag": EXCEL_DRAG_DESCRIPTION,
        "wheel_mouse_input": EXCEL_WHEEL_MOUSE_INPUT_DESCRIPTION,
        "table2markdown": EXCEL_TABLE2MARKDOWN_DESCRIPTION,
        "insert_excel_table": EXCEL_INSERT_EXCEL_TABLE_DESCRIPTION,
        "select_table_range": EXCEL_SELECT_TABLE_RANGE_DESCRIPTION,
        "set_cell_value": EXCEL_SET_CELL_VALUE_DESCRIPTION,
        "auto_fill": EXCEL_AUTO_FILL_DESCRIPTION,
        "reorder_columns": EXCEL_REORDER_COLUMNS_DESCRIPTION,
    },
    "ppt": {
        "click": PPT_CLICK_DESCRIPTION,
        "type": PPT_TYPE_DESCRIPTION,
        "drag": PPT_DRAG_DESCRIPTION,
        "wheel_mouse_input": PPT_WHEEL_MOUSE_INPUT_DESCRIPTION,
        "set_background_color": PPT_SET_BACKGROUND_COLOR_DESCRIPTION,
        "save_as": PPT_SAVE_AS_DESCRIPTION,
    },
}