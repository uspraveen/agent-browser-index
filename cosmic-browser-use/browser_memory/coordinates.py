"""Visual index helpers for MiMo-grounded browser actions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cosmic_types import ActionResult, ActionType, BrowserState, ToolCall


VISUAL_ACTIONS = {
    ActionType.VISUAL_CLICK.value,
    ActionType.VISUAL_TYPE.value,
    ActionType.VISUAL_HOVER.value,
}


def normalize_xy(x: int, y: int, width: int, height: int) -> Tuple[float, float]:
    safe_width = max(1, int(width or 1))
    safe_height = max(1, int(height or 1))
    return round(float(x) / safe_width, 6), round(float(y) / safe_height, 6)


def replay_coordinates(visual_index: Dict[str, Any], viewport_width: int, viewport_height: int) -> Optional[Tuple[int, int]]:
    if not visual_index:
        return None

    normalized = visual_index.get("normalized_coordinates") or {}
    x_norm = normalized.get("x")
    y_norm = normalized.get("y")
    if x_norm is not None and y_norm is not None:
        x = int(round(float(x_norm) * max(1, viewport_width)))
        y = int(round(float(y_norm) * max(1, viewport_height)))
        return x, y

    pixel = visual_index.get("pixel_coordinates") or {}
    if pixel.get("x") is not None and pixel.get("y") is not None:
        return int(pixel["x"]), int(pixel["y"])

    return None


def target_description_from_tool_call(tool_call: Optional[ToolCall | Dict[str, Any]]) -> str:
    if not tool_call:
        return ""
    if isinstance(tool_call, dict):
        params = tool_call.get("parameters") or {}
        action_type = str(tool_call.get("action_type") or "")
    else:
        params = tool_call.parameters or {}
        action_type = tool_call.action_type.value
    for key in ("description", "field_description", "selector", "url", "query"):
        value = params.get(key)
        if value:
            return str(value)
    return action_type


def build_visual_index(
    action: Optional[ActionResult],
    tool_call: Optional[ToolCall | Dict[str, Any]],
    before_state: Optional[BrowserState],
    screenshot_path: str,
) -> Optional[Dict[str, Any]]:
    """Build a replayable visual index from a MiMo-grounded action.

    MiMo is prompted for pixel coordinates. The browser controller already parses
    pixel or 0-1 normalized model outputs into pixels. We store both the parsed
    pixels and normalized coordinates so replay can scale to the current viewport.
    """
    if not action or not action.coordinates or not before_state:
        return None

    action_type = action.action_type.value
    if action_type not in VISUAL_ACTIONS:
        return None

    x, y = action.coordinates
    width = before_state.viewport_width
    height = before_state.viewport_height
    x_norm, y_norm = normalize_xy(x, y, width, height)
    grounding = dict((action.metadata or {}).get("mimo_grounding") or {})

    return {
        "action_type": action_type,
        "target_description": target_description_from_tool_call(tool_call) or action.description,
        "coordinate_source": "mimo_grounded",
        "screenshot_path": screenshot_path,
        "page_url": before_state.url,
        "page_title": before_state.title,
        "viewport": {"width": width, "height": height},
        "pixel_coordinates": {"x": int(x), "y": int(y)},
        "normalized_coordinates": {"x": x_norm, "y": y_norm},
        "mimo_grounding": grounding,
    }
