# copied and modified from GUI-360/evaluator/action_prediction_yr.py

from typing import Dict
from examples.ui_world_model.tool_definitions import normalize_tool_args, normalize_tool_args_a11y
import os


class ActionEvaluation:
    def __init__(self):
        self.sentence_model = None
    
    def compare_action_command(
        self,
        gt_command: str,
        pred_command: str,
    ) -> dict:
        """Compare ground truth and predicted actions, return a dictionary of results."""
        
        gt_command = self._preprocess_command(gt_command)
        pred_command = self._preprocess_command(pred_command)
        
        pred_args = pred_command.get("args", {})
        gt_args = gt_command.get("args", {})
        pred_function = pred_command.get("function", None)
        gt_function = gt_command.get("function", None)
        
        gt_rect = gt_command.get("args", {}).get("rectangle", None)
        gt_rect_end = gt_command.get("args", {}).get("rectangle_end", None)
        pred_control_name = pred_command.get("control_name", None)
        gt_control_name = gt_command.get("control_name", None)
        
        function_match = pred_function == gt_function if (pred_function and gt_function) else False
        control_name_match = pred_control_name == gt_control_name if (pred_control_name and gt_control_name) else False
        args_match = False
        
        if pred_args is not None and gt_args is not None:
            if pred_function == "drag" and gt_function == "drag":
                # Special comparison for drag operations with rectangle-based matching
                args_match = self._compare_drag_args(
                    pred_args, gt_args, gt_rect, gt_rect_end
                )
            else:
                # Regular comparison for other operations
                args_match = self._compare_regular_args(
                    pred_args, gt_args, gt_rect, pred_function, gt_function
                )
        return {
            "function_match": function_match,
            "control_name_match": control_name_match,
            "args_match": args_match,
            "overall_match": function_match and control_name_match and args_match,
        }
    
    def compare_action_description(
        self,
        gt_description: str,
        pred_description: str,
    ) -> float:
        self._load_sentence_model()
        action_desc_embeddings = self.sentence_model.encode(
            [gt_description, pred_description]
        )
        action_desc_sim = self.sentence_model.similarity(
            action_desc_embeddings[0],
            action_desc_embeddings[1],
        )
        return action_desc_sim.item()
        
    def _load_sentence_model(self):
        if self.sentence_model is None:
            from sentence_transformers import SentenceTransformer, util
            self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def _preprocess_command(self, action: dict, resolution=(1024, 736)) -> dict:
        print("Original action:", action)
        if action["function"] == "drag":
            start_x = action["args"]["start_x"]
            start_y = action["args"]["start_y"]
            end_x = action["args"]["end_x"]
            end_y = action["args"]["end_y"]

            action["args"]["start_coordinate"] = [start_x, start_y]
            action["args"]["end_coordinate"] = [end_x, end_y]

            action["args"]["rectangle"] = {
                "left": max(0, start_x) - 25,
                "top": max(0, start_y) - 25,
                "right": min(start_x + 25, resolution[0]),
                "bottom": min(start_y + 25, resolution[1]),
            }
            
            action["args"]["rectangle_end"] = {
                "left": max(0, end_x) - 25,
                "top": max(0, end_y) - 25,
                "right": min(end_x + 25, resolution[0]),
                "bottom": min(end_y + 25, resolution[1]),
            }
        else:
            if "coordinate_x" in action and action["coordinate_x"]:
                action["args"]["coordinate"] = [
                    action["coordinate_x"],
                    action["coordinate_y"],
                ]
        return action

    def _compare_drag_args(
        self,
        pred_args: Dict,
        gt_args: Dict,
        gt_rect: Dict = None,
        gt_rect_end: Dict = None,
    ) -> bool:
        """Compare drag operation arguments with rectangle-based matching.

        For drag operations, predicted coordinates are considered correct if they fall within
        the ground truth rectangles for start and end positions.

        Args:
            pred_args: Predicted arguments containing start_coordinate and end_coordinate
            gt_args: Ground truth arguments containing start_coordinate and end_coordinate
            gt_rect: Ground truth rectangle for start coordinate
            gt_rect_end: Ground truth rectangle for end coordinate

        Returns:
            bool: True if drag arguments match (coordinates within rectangles)
        """
        try:
            # Normalize both argument sets with default values
            pred_normalized = normalize_tool_args("drag", pred_args)
            gt_normalized = normalize_tool_args("drag", gt_args)

            # Check if both have required drag coordinates
            if (
                "start_coordinate" not in pred_normalized
                or "end_coordinate" not in pred_normalized
            ):
                return False
            if (
                "start_coordinate" not in gt_normalized
                or "end_coordinate" not in gt_normalized
            ):
                return False

            pred_start = pred_normalized["start_coordinate"]
            pred_end = pred_normalized["end_coordinate"]
            gt_start = gt_normalized["start_coordinate"]
            gt_end = gt_normalized["end_coordinate"]

            # Ensure coordinates are lists/tuples with 2 elements
            if not (isinstance(pred_start, (list, tuple)) and len(pred_start) == 2):
                return False
            if not (isinstance(pred_end, (list, tuple)) and len(pred_end) == 2):
                return False
            if not (isinstance(gt_start, (list, tuple)) and len(gt_start) == 2):
                return False
            if not (isinstance(gt_end, (list, tuple)) and len(gt_end) == 2):
                return False

            # Check if predicted coordinates fall within ground truth rectangles
            start_match = True
            end_match = True

            if gt_rect:
                # Check if predicted start coordinate is within ground truth start rectangle
                pred_start_x, pred_start_y = float(pred_start[0]), float(pred_start[1])
                start_match = (
                    gt_rect["left"] <= pred_start_x <= gt_rect["right"]
                    and gt_rect["top"] <= pred_start_y <= gt_rect["bottom"]
                )
            else:
                # Fall back to tolerance-based comparison if no rectangle provided
                tolerance = 25.0  # Default tolerance
                start_match = (
                    abs(float(pred_start[0]) - float(gt_start[0])) <= tolerance
                    and abs(float(pred_start[1]) - float(gt_start[1])) <= tolerance
                )

            if gt_rect_end:
                # Check if predicted end coordinate is within ground truth end rectangle
                pred_end_x, pred_end_y = float(pred_end[0]), float(pred_end[1])
                end_match = (
                    gt_rect_end["left"] <= pred_end_x <= gt_rect_end["right"]
                    and gt_rect_end["top"] <= pred_end_y <= gt_rect_end["bottom"]
                )
            else:
                # Fall back to tolerance-based comparison if no rectangle provided
                tolerance = 25.0  # Default tolerance
                end_match = (
                    abs(float(pred_end[0]) - float(gt_end[0])) <= tolerance
                    and abs(float(pred_end[1]) - float(gt_end[1])) <= tolerance
                )

            coordinates_match = start_match and end_match

            # Compare other arguments using normalized values
            # Exclude coordinates from comparison as they are handled separately
            other_args_match = True
            for key in ["button", "duration", "key_hold"]:
                pred_val = pred_normalized.get(key)
                gt_val = gt_normalized.get(key)

                # Convert to strings for comparison, handling None values
                pred_str = str(pred_val).lower() if pred_val is not None else "none"
                gt_str = str(gt_val).lower() if gt_val is not None else "none"

                if pred_str != gt_str:
                    other_args_match = False
                    break

            return coordinates_match and other_args_match

        except (ValueError, TypeError, KeyError) as e:
            print(f"Error comparing drag arguments: {e}")
            return False

    def _compare_regular_args(
        self,
        pred_args: Dict,
        gt_args: Dict,
        gt_rect: Dict = None,
        pred_function: str = None,
        gt_function: str = None,
    ) -> bool:
        """Compare regular (non-drag) operation arguments.

        For coordinate-based operations, check if predicted coordinates fall within ground truth rectangle.
        Uses normalized arguments with default values.

        Args:
            pred_args: Predicted arguments
            gt_args: Ground truth arguments
            gt_rect: Ground truth rectangle for coordinate validation
            pred_function: Predicted function name for normalization
            gt_function: Ground truth function name for normalization

        Returns:
            bool: True if arguments match
        """
        try:
            # If we don't have function names, try to infer from the presence of coordinate types
            if pred_function is None:
                if "coordinate" in pred_args:
                    pred_function = (
                        "click"  # Default assumption for coordinate-based operations
                    )
                else:
                    pred_function = "unknown"

            if gt_function is None:
                if "coordinate" in gt_args:
                    gt_function = (
                        "click"  # Default assumption for coordinate-based operations
                    )
                else:
                    gt_function = "unknown"

            # Normalize both argument sets with default values
            pred_normalized = normalize_tool_args(pred_function, pred_args)
            gt_normalized = normalize_tool_args(gt_function, gt_args)

            # Special handling for coordinate arguments
            if "coordinate" in pred_normalized and "coordinate" in gt_normalized:
                pred_coord = pred_normalized["coordinate"]
                gt_coord = gt_normalized["coordinate"]

                # Ensure coordinates are lists/tuples with 2 elements
                if (
                    isinstance(pred_coord, (list, tuple))
                    and len(pred_coord) == 2
                    and isinstance(gt_coord, (list, tuple))
                    and len(gt_coord) == 2
                ):

                    if gt_rect:
                        # Check if predicted coordinate is within ground truth rectangle
                        pred_x, pred_y = float(pred_coord[0]), float(pred_coord[1])
                        coordinate_match = (
                            gt_rect["left"] <= pred_x <= gt_rect["right"]
                            and gt_rect["top"] <= pred_y <= gt_rect["bottom"]
                        )
                    else:
                        # Fall back to tolerance-based comparison if no rectangle provided
                        tolerance = 25.0  # Default tolerance
                        coordinate_match = (
                            abs(float(pred_coord[0]) - float(gt_coord[0])) <= tolerance
                            and abs(float(pred_coord[1]) - float(gt_coord[1]))
                            <= tolerance
                        )

                    # Compare other arguments using normalized values (excluding coordinate)
                    other_args_match = True
                    for key in pred_normalized:
                        if key != "coordinate":
                            pred_val = pred_normalized[key]
                            gt_val = gt_normalized.get(key)

                            # Convert to strings for comparison, handling None values
                            pred_str = (
                                str(pred_val).lower()
                                if pred_val is not None
                                else "none"
                            )
                            gt_str = (
                                str(gt_val).lower() if gt_val is not None else "none"
                            )

                            if pred_str != gt_str:
                                other_args_match = False
                                break

                    return coordinate_match and other_args_match

            # Standard comparison for non-coordinate operations using normalized values
            pred_normalized_str = {}
            gt_normalized_str = {}

            for k, v in pred_normalized.items():
                if isinstance(v, (str, bool)):
                    pred_normalized_str[k] = str(v).lower()
                elif isinstance(v, (list, tuple)):
                    pred_normalized_str[k] = v
                elif v is None:
                    pred_normalized_str[k] = "none"
                else:
                    pred_normalized_str[k] = v

            for k, v in gt_normalized.items():
                if isinstance(v, (str, bool)):
                    gt_normalized_str[k] = str(v).lower()
                elif isinstance(v, (list, tuple)):
                    gt_normalized_str[k] = v
                elif v is None:
                    gt_normalized_str[k] = "none"
                else:
                    gt_normalized_str[k] = v

            return pred_normalized_str == gt_normalized_str

        except Exception as e:
            print(f"Error comparing regular arguments: {e}")
            return False

class ActionEvaluationA11y:
    def __init__(self):
        pass
    
    def compare_action_command(
        self,
        gt_raw: dict,
        gt_command: dict,
        pred_command: dict,
        resolution=(1024, 736),
    ) -> dict:
        # print(gt_command)
        # print(pred_command)
        import pdb; pdb.set_trace()
        gt_function, pred_function = gt_command["function"], pred_command["function"]
        gt_args, pred_args = gt_command["args"], pred_command["args"]
        gt_status, pred_status = gt_command["status"], pred_command["status"]
        
        gt_rect = gt_raw.get("rectangle", {})
        
        if gt_function == "drag":
            start_x = gt_raw["args"]["start_x"]
            start_y = gt_raw["args"]["start_y"]
            end_x = gt_raw["args"]["end_x"]
            end_y = gt_raw["args"]["end_y"]
            gt_rect = {
                "left": max(0, start_x - 25),
                "top": max(0, start_y - 25),
                "right": min(start_x + 25, resolution[0]),
                "bottom": min(start_y + 25, resolution[1]),
            }
            gt_rect_end = {
                "left": max(0, end_x - 25),
                "top": max(0, end_y - 25),
                "right": min(end_x + 25, resolution[0]),
                "bottom": min(end_y + 25, resolution[1]),
            }
        
        function_match = gt_function == pred_function if (gt_function and pred_function) else False
        status_match = pred_status == gt_status if pred_status else False
        
        args_match = False
        if pred_args is not None and gt_args is not None:
            if pred_function == "drag" and gt_function == "drag":
                # Special comparison for drag operations with rectangle-based matching
                args_match = self._compare_drag_args(
                    pred_args, gt_args, gt_rect, gt_rect_end
                )
            else:
                # Enhanced comparison for A11Y operations
                args_match = self._compare_a11y_args(
                    pred_args, gt_args, gt_rect, pred_function, gt_function
                )
        return {
            "function_match": function_match,
            "status_match": status_match,
            "args_match": args_match,
            "overall_match": function_match and status_match and args_match,
        }
        
    def compare_action_command_2_pred(
        self,
        gt_raw: dict,
        gt_command: dict,
        pred_command: dict,
    ) -> dict:
        """
        Only supports the case where `gt_raw` is empty / not provided.
        Compare two predicted commands (gt_command vs pred_command) without any rectangle from raw GT.
        """
        gt_function = gt_command.get("function", None)
        pred_function = pred_command.get("function", None)

        gt_args = gt_command.get("args", {}) or {}
        pred_args = pred_command.get("args", {}) or {}

        gt_status = gt_command.get("status", None)
        pred_status = pred_command.get("status", None)

        # No gt_raw: no rectangle-based matching available
        gt_rect = None
        gt_rect_end = None

        function_match = (gt_function == pred_function) if (gt_function and pred_function) else False
        status_match = (pred_status == gt_status) if (pred_status is not None and gt_status is not None) else False

        args_match = False
        if pred_args is not None and gt_args is not None:
            if pred_function == "drag" and gt_function == "drag":
                # Will fall back to tolerance-based matching since gt_rect/gt_rect_end are None
                args_match = self._compare_drag_args(pred_args, gt_args, gt_rect, gt_rect_end)
            else:
                # A11Y arg comparison without gt_rect (falls back to coordinate-to-coordinate tolerance if needed)
                args_match = self._compare_a11y_args(pred_args, gt_args, gt_rect, pred_function, gt_function)

        return {
            "function_match": function_match,
            "status_match": status_match,
            "args_match": args_match,
            "overall_match": function_match and status_match and args_match,
        }

    

    def _compare_drag_args(
        self,
        pred_args: Dict,
        gt_args: Dict,
        gt_rect: Dict = None,
        gt_rect_end: Dict = None,
    ) -> bool:
        """Compare drag operation arguments with rectangle-based matching."""
        try:
            # Normalize both argument sets with A11Y default values
            pred_normalized = normalize_tool_args_a11y("drag", pred_args)
            gt_normalized = normalize_tool_args_a11y("drag", gt_args)

            # Check if both have required drag coordinates
            if (
                "start_coordinate" not in pred_normalized
                or "end_coordinate" not in pred_normalized
            ):
                return False
            if (
                "start_coordinate" not in gt_normalized
                or "end_coordinate" not in gt_normalized
            ):
                return False

            pred_start = pred_normalized["start_coordinate"]
            pred_end = pred_normalized["end_coordinate"]
            gt_start = gt_normalized["start_coordinate"]
            gt_end = gt_normalized["end_coordinate"]

            # Ensure coordinates are lists/tuples with 2 elements
            if not (isinstance(pred_start, (list, tuple)) and len(pred_start) == 2):
                return False
            if not (isinstance(pred_end, (list, tuple)) and len(pred_end) == 2):
                return False
            if not (isinstance(gt_start, (list, tuple)) and len(gt_start) == 2):
                return False
            if not (isinstance(gt_end, (list, tuple)) and len(gt_end) == 2):
                return False

            # Check if predicted coordinates fall within ground truth rectangles
            start_match = True
            end_match = True

            if gt_rect:
                # Check if predicted start coordinate is within ground truth start rectangle
                pred_start_x, pred_start_y = float(pred_start[0]), float(pred_start[1])
                start_match = (
                    gt_rect["left"] <= pred_start_x <= gt_rect["right"]
                    and gt_rect["top"] <= pred_start_y <= gt_rect["bottom"]
                )
            else:
                # Fall back to tolerance-based comparison if no rectangle provided
                tolerance = 25.0  # Default tolerance
                start_match = (
                    abs(float(pred_start[0]) - float(gt_start[0])) <= tolerance
                    and abs(float(pred_start[1]) - float(gt_start[1])) <= tolerance
                )

            if gt_rect_end:
                # Check if predicted end coordinate is within ground truth end rectangle
                pred_end_x, pred_end_y = float(pred_end[0]), float(pred_end[1])
                end_match = (
                    gt_rect_end["left"] <= pred_end_x <= gt_rect_end["right"]
                    and gt_rect_end["top"] <= pred_end_y <= gt_rect_end["bottom"]
                )
            else:
                # Fall back to tolerance-based comparison if no rectangle provided
                tolerance = 25.0  # Default tolerance
                end_match = (
                    abs(float(pred_end[0]) - float(gt_end[0])) <= tolerance
                    and abs(float(pred_end[1]) - float(gt_end[1])) <= tolerance
                )

            coordinates_match = start_match and end_match

            # Compare other arguments using normalized values
            # Exclude coordinates from comparison as they are handled separately
            other_args_match = True
            for key in ["button", "duration", "key_hold"]:
                pred_val = pred_normalized.get(key)
                gt_val = gt_normalized.get(key)

                # Convert to strings for comparison, handling None values
                pred_str = str(pred_val).lower() if pred_val is not None else "none"
                gt_str = str(gt_val).lower() if gt_val is not None else "none"

                if pred_str != gt_str:
                    other_args_match = False
                    break

            return coordinates_match and other_args_match

        except (ValueError, TypeError, KeyError) as e:
            print(f"Error comparing drag arguments: {e}")
            return False
        
    def _compare_a11y_args(
        self,
        pred_args: Dict,
        gt_args: Dict,
        gt_rect: Dict = None,
        pred_function: str = None,
        gt_function: str = None,
    ) -> bool:
        """Compare A11Y operation arguments based on model output type.
        
        Prioritizes comparison based on what the model outputs:
        - If model outputs control_label, use gt control_label for comparison
        - If model outputs coordinate, use gt coordinate/rectangle for comparison
        - Both ground truth rect and control_label are available for comparison
        """
        try:
            # If we don't have function names, try to infer from the presence of argument types
            if pred_function is None:
                if "control_label" in pred_args or "coordinate" in pred_args:
                    pred_function = "click"  # Default assumption
                else:
                    pred_function = "unknown"

            if gt_function is None:
                if "control_label" in gt_args or "coordinate" in gt_args:
                    gt_function = "click"  # Default assumption
                else:
                    gt_function = "unknown"

            # Normalize both argument sets with A11Y default values
            pred_normalized = normalize_tool_args_a11y(pred_function, pred_args)
            gt_normalized = normalize_tool_args_a11y(gt_function, gt_args)

            # Priority based on model output: check what the model actually predicted
            model_has_control_label = "control_label" in pred_normalized and pred_normalized["control_label"] is not None
            model_has_coordinate = "coordinate" in pred_normalized and pred_normalized["coordinate"] is not None
            
            # Case 1: Model outputs control_label - prioritize control_label comparison
            if model_has_control_label:
                if "control_label" in gt_normalized and gt_normalized["control_label"] is not None:
                    pred_label = pred_normalized["control_label"]
                    gt_label = gt_normalized["control_label"]
                    
                    control_label_match = str(pred_label) == str(gt_label)
                    
                    # If control labels match, check other non-coordinate arguments
                    if control_label_match:
                        other_args_match = True
                        for key in pred_normalized:
                            if key not in ["control_label", "coordinate"]:
                                pred_val = pred_normalized[key]
                                gt_val = gt_normalized.get(key)

                                # Convert to strings for comparison, handling None values
                                pred_str = (
                                    str(pred_val).lower()
                                    if pred_val is not None
                                    else "none"
                                )
                                gt_str = (
                                    str(gt_val).lower() if gt_val is not None else "none"
                                )

                                if pred_str != gt_str:
                                    other_args_match = False
                                    break

                        return control_label_match and other_args_match
                    else:
                        return False
                else:
                    # Model predicted control_label but GT doesn't have it - fallback to coordinate if available
                    if model_has_coordinate and ("coordinate" in gt_normalized or gt_rect):
                        return self._compare_coordinate_args(pred_normalized, gt_normalized, gt_rect)
                    else:
                        return False
            
            # Case 2: Model outputs coordinate - prioritize coordinate comparison  
            elif model_has_coordinate:
                return self._compare_coordinate_args(pred_normalized, gt_normalized, gt_rect)
            
            # Case 3: Standard comparison for non-coordinate/non-control_label operations
            else:
                return self._compare_standard_args(pred_normalized, gt_normalized)

        except Exception as e:
            print(f"Error comparing A11Y arguments: {e}")
            return False

    
    def _compare_coordinate_args(self, pred_normalized: Dict, gt_normalized: Dict, gt_rect: Dict = None) -> bool:
        """Helper method to compare coordinate-based arguments."""
        if "coordinate" not in pred_normalized:
            return False
            
        pred_coord = pred_normalized["coordinate"]
        
        # Ensure predicted coordinate is valid
        if not (isinstance(pred_coord, (list, tuple)) and len(pred_coord) == 2):
            return False

        coordinate_match = False
        
        # Try rectangle-based comparison first if available
        if gt_rect:
            pred_x, pred_y = float(pred_coord[0]), float(pred_coord[1])
            coordinate_match = (
                gt_rect["left"] <= pred_x <= gt_rect["right"]
                and gt_rect["top"] <= pred_y <= gt_rect["bottom"]
            )
        # Fallback to coordinate-to-coordinate comparison if GT has coordinate
        elif "coordinate" in gt_normalized and gt_normalized["coordinate"] is not None:
            gt_coord = gt_normalized["coordinate"]
            if isinstance(gt_coord, (list, tuple)) and len(gt_coord) == 2:
                tolerance = 25.0  # Default tolerance
                coordinate_match = (
                    abs(float(pred_coord[0]) - float(gt_coord[0])) <= tolerance
                    and abs(float(pred_coord[1]) - float(gt_coord[1])) <= tolerance
                )
        
        if not coordinate_match:
            return False
            
        # Compare other arguments using normalized values (excluding coordinate and control_label)
        other_args_match = True
        for key in pred_normalized:
            if key not in ["coordinate", "control_label"]:
                pred_val = pred_normalized[key]
                gt_val = gt_normalized.get(key)

                # Convert to strings for comparison, handling None values
                pred_str = str(pred_val).lower() if pred_val is not None else "none"
                gt_str = str(gt_val).lower() if gt_val is not None else "none"

                if pred_str != gt_str:
                    other_args_match = False
                    break

        return coordinate_match and other_args_match

    
    def _compare_standard_args(self, pred_normalized: Dict, gt_normalized: Dict) -> bool:
        """Helper method for standard argument comparison."""
        pred_normalized_str = {}
        gt_normalized_str = {}

        for k, v in pred_normalized.items():
            if isinstance(v, (str, bool)):
                pred_normalized_str[k] = str(v).lower()
            elif isinstance(v, (list, tuple)):
                pred_normalized_str[k] = v
            elif v is None:
                pred_normalized_str[k] = "none"
            else:
                pred_normalized_str[k] = v

        for k, v in gt_normalized.items():
            if isinstance(v, (str, bool)):
                gt_normalized_str[k] = str(v).lower()
            elif isinstance(v, (list, tuple)):
                gt_normalized_str[k] = v
            elif v is None:
                gt_normalized_str[k] = "none"
            else:
                gt_normalized_str[k] = v

        return pred_normalized_str == gt_normalized_str