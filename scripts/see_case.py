import json
import logging
import os

logger = logging.getLogger(__name__)

JSON_PATH = os.environ.get("EVAL_RESULTS_PATH", "data/agent_eval_result/eval_result/agent_wm_evaluation_results.json")

NONE_KEY = "action_selection_none.json"
TEXT_KEY = "action_selection_text.json"
BASE_IMAGE_KEY = "generated_base/action_selection_image.json"
BASE_TEXT_IMAGE_KEY = "generated_base/action_selection_text+image.json"
E24_IMAGE_KEY = "generated_epoch-24/action_selection_image.json"
E24_TEXT_IMAGE_KEY = "generated_epoch-24/action_selection_text+image.json"


with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

none_result = data[NONE_KEY]
text_result = data[TEXT_KEY]
base_image_result = data[BASE_IMAGE_KEY]
base_text_image_result = data[BASE_TEXT_IMAGE_KEY]
e24_image_result = data[E24_IMAGE_KEY]
e24_text_image_result = data[E24_TEXT_IMAGE_KEY]

# 找 case：base 对，epoch-24 错
selected_cases = []

for case_path, base_info in base_text_image_result.items():
    base_ok = base_info.get("overall_match", False)
    epoch24_ok = e24_text_image_result.get(case_path, {}).get("overall_match", False)

    if base_ok and not epoch24_ok:
        selected_cases.append(case_path)

logger.info("Found %d cases:", len(selected_cases))

for c in selected_cases:
    logger.info("CASE: %s", c)
    logger.info("  none: %s", none_result[c])
    logger.info("  text: %s", text_result[c])
    logger.info("  base image: %s", base_image_result[c])
    logger.info("  epoch-24: %s", e24_image_result.get(c))
    logger.info("  base text+image: %s", base_text_image_result[c])
    logger.info("  epoch-24 text+image: %s", e24_text_image_result.get(c))