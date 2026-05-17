#!/usr/bin/env python3
"""
MiMo-VL Coordinate Finder for Cosmic Browser Use Agent
========================================================
Finds UI element coordinates using the MiMo-VL-7B-RL vision model.

MiMo-VL has exceptional GUI understanding and grounding capabilities,
outperforming specialized models like UI-TARS on benchmarks like OSWorld-G.

Key Features:
- Thinking mode (default): Full reasoning process with chain-of-thought
- No-think mode: Direct responses without reasoning (faster)
- Trained with RL on grounding tasks for precise coordinate prediction

Usage:
    python find_coordinates_mimo.py <image_path> "instruction" [options]

Examples:
    # With reasoning (default)
    python find_coordinates_mimo.py screen.png "click the search box"
    
    # Without reasoning (faster, for simple tasks)
    python find_coordinates_mimo.py screen.png "search icon" --no-think
    
    # With annotation
    python find_coordinates_mimo.py screen.png "Log In button" --annotate
"""

import argparse
import ast
import base64
import json
import os
import re
import sys
from typing import Tuple, Dict, Any, Optional, List

import requests
from PIL import Image, ImageDraw

# Configuration - can be overridden via environment variables or args
DEFAULT_VLLM_URL = os.environ.get("MIMO_API_URL", "http://cosmos-9.ddns.ualr.edu:8098/v1/chat/completions")
DEFAULT_MODEL_ID = os.environ.get("MIMO_MODEL_ID", "XiaomiMiMo/MiMo-VL-7B-RL")
DEFAULT_TIMEOUT = int(os.environ.get("MIMO_TIMEOUT", "180"))  # Longer timeout for reasoning
DEFAULT_API_KEY = os.environ.get("MIMO_API_KEY", "")

# =============================================================================
# MiMo-VL Prompts for GUI Grounding
# =============================================================================

# Grounding prompt - asks for click coordinates
# MiMo-VL uses pixel coordinates (not normalized) based on Qwen2.5-VL training
GROUNDING_SYSTEM_PROMPT = """You are a GUI grounding assistant. Given a screenshot and an element description, output the pixel coordinates (x, y) of the center of that element.

Output format: Return ONLY the coordinates as [x, y] where x and y are pixel values.
Do not include any other text, explanation, or formatting."""

# Navigation prompt for action-aware grounding (agentic mode)
NAV_SYSTEM_PROMPT = """You are a GUI agent assistant. Given a screenshot and a task instruction, determine the next action and its target coordinates.

Output format: Return a JSON object with:
- "action": one of CLICK, INPUT, SCROLL, HOVER, ENTER
- "value": text to input (for INPUT action) or scroll direction (for SCROLL), null otherwise
- "position": [x, y] pixel coordinates of the target element, or null if not applicable

Example: {"action": "CLICK", "value": null, "position": [500, 300]}"""


def image_size(path: str) -> Tuple[int, int]:
    """Get image dimensions (width, height)."""
    with Image.open(path) as im:
        return im.size


def image_to_data_url(path: str) -> str:
    """Convert image file to base64 data URL."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = path.lower().rsplit(".", 1)[-1]
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
    return f"data:{mime};base64,{b64}"


def call_mimo_grounding(
    image_data_url: str,
    element_query: str,
    image_size: Tuple[int, int],
    api_url: str = DEFAULT_VLLM_URL,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 1024,
    timeout: int = DEFAULT_TIMEOUT,
    no_think: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.95,
    api_key: str = DEFAULT_API_KEY,
) -> str:
    """
    Call MiMo-VL in grounding mode.
    
    Args:
        image_data_url: Base64 encoded image
        element_query: Description of element to find
        image_size: (width, height) of the image
        api_url: vLLM API endpoint
        model_id: Model identifier
        max_tokens: Maximum tokens to generate
        timeout: Request timeout
        no_think: If True, append /no_think to disable reasoning
        temperature: Sampling temperature (recommended: 0.3)
        top_p: Top-p sampling (recommended: 0.95)
    
    Returns:
        Raw model output string
    """
    w, h = image_size
    
    # Build the query with image dimensions context
    query = f"""Image size: {w}x{h} pixels

Find the element: {element_query}

Output the center coordinates as [x, y] in pixels."""

    # Append /no_think to disable reasoning mode
    if no_think:
        query += " /no_think"
    
    # MiMo-VL requires image BEFORE text for single image inputs
    messages = [
        {
            "role": "system",
            "content": GROUNDING_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": query},
            ],
        }
    ]

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_mimo_navigation(
    image_data_url: str,
    task: str,
    image_size: Tuple[int, int],
    action_history: Optional[List[str]] = None,
    api_url: str = DEFAULT_VLLM_URL,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 1024,
    timeout: int = DEFAULT_TIMEOUT,
    no_think: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.95,
    api_key: str = DEFAULT_API_KEY,
) -> str:
    """
    Call MiMo-VL in navigation/agentic mode.
    
    Returns action dict with coordinates.
    """
    w, h = image_size
    
    query = f"""Image size: {w}x{h} pixels
Task: {task}"""
    
    if action_history:
        query += f"\nPrevious actions: {' -> '.join(action_history)}"
    
    query += "\n\nDetermine the next action and output as JSON."
    
    if no_think:
        query += " /no_think"
    
    messages = [
        {
            "role": "system", 
            "content": NAV_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": query},
            ],
        }
    ]

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def check_mimo_health(api_url: str = DEFAULT_VLLM_URL, timeout: int = 5, api_key: str = DEFAULT_API_KEY) -> bool:
    """
    Check if MiMo API is reachable and healthy.
    
    Tries to hit the /v1/models endpoint to verify connectivity.
    """
    try:
        # Construct models endpoint from chat endpoint or base URL
        # Typical url: http://host:port/v1/chat/completions
        # Target url: http://host:port/v1/models
        
        base_url = api_url
        if "/chat/completions" in base_url:
            base_url = base_url.replace("/chat/completions", "/models")
        elif "/v1" not in base_url:
            base_url = f"{base_url.rstrip('/')}/v1/models"
        else:
            # Assume it might be just base, try appending /models if not present
            if not base_url.endswith("/models"):
                 base_url = f"{base_url.rstrip('/')}/models"
                 
        # fallback: just try the base url if simple logic failed
        print(f"   (Pinging MiMo at: {base_url})")
            
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.get(base_url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        print(f"   (Ping failed: {e})")
        return False


def extract_thinking(raw: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking/reasoning content from model output.
    
    MiMo-VL wraps reasoning in <think>...</think> tags.
    
    Returns:
        (thinking_content, remaining_output)
    """
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        remaining = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        return thinking, remaining
    return None, raw


def parse_coordinates(raw: str, image_size: Tuple[int, int]) -> Tuple[int, int, bool]:
    """
    Parse coordinates from model output.
    
    MiMo-VL typically outputs pixel coordinates directly.
    Also handles normalized coordinates (0-1 range) as fallback.
    
    Args:
        raw: Model output string
        image_size: (width, height) for normalization conversion
    
    Returns:
        (x_px, y_px, is_normalized) where is_normalized indicates if conversion was needed
    
    Raises:
        ValueError if parsing fails
    """
    w, h = image_size
    
    # Remove thinking tags if present
    _, clean = extract_thinking(raw)
    
    # Try to find [x, y] pattern (most common)
    list_match = re.search(r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]', clean)
    if list_match:
        x, y = float(list_match.group(1)), float(list_match.group(2))
        # Check if normalized (0-1 range) or pixel coordinates
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0:
            # Could be normalized - check if it makes sense as pixels
            if x < 2 and y < 2:  # Almost certainly normalized
                return int(round(x * w)), int(round(y * h)), True
        # Pixel coordinates
        return int(round(x)), int(round(y)), False
    
    # Try (x, y) pattern
    tuple_match = re.search(r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', clean)
    if tuple_match:
        x, y = float(tuple_match.group(1)), float(tuple_match.group(2))
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h)), True
        return int(round(x)), int(round(y)), False
    
    # Try x=... y=... pattern
    xy_match = re.search(r'x\s*[=:]\s*(\d+(?:\.\d+)?)[,\s]+y\s*[=:]\s*(\d+(?:\.\d+)?)', clean, re.I)
    if xy_match:
        x, y = float(xy_match.group(1)), float(xy_match.group(2))
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h)), True
        return int(round(x)), int(round(y)), False
    
    # Try JSON with position field
    json_match = re.search(r'"position"\s*:\s*\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]', clean)
    if json_match:
        x, y = float(json_match.group(1)), float(json_match.group(2))
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h)), True
        return int(round(x)), int(round(y)), False
    
    # Try to find any two numbers that could be coordinates
    numbers = re.findall(r'(\d+(?:\.\d+)?)', clean)
    if len(numbers) >= 2:
        x, y = float(numbers[0]), float(numbers[1])
        # Sanity check - coordinates should be within image bounds (with some margin)
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h)), True
        elif 0 <= x <= w * 1.1 and 0 <= y <= h * 1.1:
            return int(round(min(x, w))), int(round(min(y, h))), False
    
    raise ValueError(f"Could not parse coordinates from: {raw!r}")


def parse_navigation_output(raw: str, image_size: Tuple[int, int]) -> Dict[str, Any]:
    """
    Parse navigation mode output into structured action dict.
    
    Returns dict with action, value, position, position_px
    """
    _, clean = extract_thinking(raw)
    w, h = image_size
    
    # Try to extract JSON
    json_match = re.search(r'\{[^{}]+\}', clean)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if 'action' in parsed:
                # Convert position if present
                if parsed.get('position'):
                    pos = parsed['position']
                    if isinstance(pos, list) and len(pos) == 2:
                        x, y = pos
                        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
                            parsed['position_px'] = [int(round(x * w)), int(round(y * h))]
                            parsed['was_normalized'] = True
                        else:
                            parsed['position_px'] = [int(round(x)), int(round(y))]
                            parsed['was_normalized'] = False
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Try ast.literal_eval
    try:
        parsed = ast.literal_eval(clean)
        if isinstance(parsed, dict) and 'action' in parsed:
            if parsed.get('position'):
                pos = parsed['position']
                if isinstance(pos, list) and len(pos) == 2:
                    x, y = pos
                    if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
                        parsed['position_px'] = [int(round(x * w)), int(round(y * h))]
                        parsed['was_normalized'] = True
                    else:
                        parsed['position_px'] = [int(round(x)), int(round(y))]
                        parsed['was_normalized'] = False
            return parsed
    except (ValueError, SyntaxError):
        pass
    
    raise ValueError(f"Could not parse navigation action from: {raw!r}")


def clamp(v: int, min_v: int, max_v: int) -> int:
    """Clamp value to range."""
    return max(min_v, min(max_v, v))


def annotate_image(
    image_path: str, 
    x_px: int, 
    y_px: int, 
    output_path: Optional[str] = None,
    marker_radius: int = 12,
    marker_color: str = "lime",
    marker_width: int = 3
) -> str:
    """
    Annotate image with a crosshair marker at the specified coordinates.
    Uses lime green for better visibility on various backgrounds.
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_mimo_annotated{ext}"
    
    with Image.open(image_path) as im:
        draw = ImageDraw.Draw(im)
        r = marker_radius
        
        # Outer circle (dark outline for contrast)
        draw.ellipse(
            (x_px - r - 1, y_px - r - 1, x_px + r + 1, y_px + r + 1), 
            outline="black", 
            width=marker_width + 2
        )
        
        # Main circle
        draw.ellipse(
            (x_px - r, y_px - r, x_px + r, y_px + r), 
            outline=marker_color, 
            width=marker_width
        )
        
        # Crosshair with dark outline
        for offset in [-1, 1]:
            draw.line((x_px - r * 2, y_px + offset, x_px + r * 2, y_px + offset), fill="black", width=1)
            draw.line((x_px + offset, y_px - r * 2, x_px + offset, y_px + r * 2), fill="black", width=1)
        
        draw.line((x_px - r * 2, y_px, x_px + r * 2, y_px), fill=marker_color, width=marker_width - 1)
        draw.line((x_px, y_px - r * 2, x_px, y_px + r * 2), fill=marker_color, width=marker_width - 1)
        
        # Center dot
        draw.ellipse(
            (x_px - 2, y_px - 2, x_px + 2, y_px + 2),
            fill=marker_color
        )
        
        im.save(output_path)
    
    return output_path


def find_coordinates(
    image_path: str,
    instruction: str,
    api_url: str = DEFAULT_VLLM_URL,
    model_id: str = DEFAULT_MODEL_ID,
    timeout: int = DEFAULT_TIMEOUT,
    annotate: bool = False,
    annotate_output: Optional[str] = None,
    no_think: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.95,
    api_key: str = DEFAULT_API_KEY,
) -> Dict[str, Any]:
    """
    Find element coordinates using MiMo-VL grounding mode.
    
    Args:
        image_path: Path to the screenshot/image file
        instruction: Element description (e.g., "search box", "Log In button")
        api_url: vLLM API endpoint URL
        model_id: Model identifier
        timeout: API request timeout in seconds
        annotate: Whether to create an annotated image
        annotate_output: Custom path for annotated image
        no_think: Disable reasoning mode for faster responses
        temperature: Sampling temperature
        top_p: Top-p sampling
        
    Returns:
        Dictionary with coordinates and metadata
    """
    # Get image dimensions
    try:
        w_img, h_img = image_size(image_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to read image: {e}"}
    
    # Convert to data URL
    try:
        img_url = image_to_data_url(image_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to encode image: {e}"}
    
    # Call MiMo-VL model
    try:
        raw_output = call_mimo_grounding(
            img_url, instruction, (w_img, h_img),
            api_url, model_id, timeout=timeout,
            no_think=no_think, temperature=temperature, top_p=top_p, api_key=api_key
        )
    except requests.RequestException as e:
        return {"status": "error", "error": f"API request failed: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error during inference: {e}"}
    
    # Extract thinking if present
    thinking, clean_output = extract_thinking(raw_output)
    
    # Parse coordinates
    try:
        x_px, y_px, was_normalized = parse_coordinates(raw_output, (w_img, h_img))
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e),
            "raw_model_output": raw_output,
            "thinking": thinking,
        }
    
    # Clamp to image bounds
    x_px = clamp(x_px, 0, w_img - 1)
    y_px = clamp(y_px, 0, h_img - 1)
    
    result = {
        "status": "success",
        "instruction": instruction,
        "image_path": image_path,
        "image_size": [w_img, h_img],
        "x_px": x_px,
        "y_px": y_px,
        "x_norm": round(x_px / w_img, 4),
        "y_norm": round(y_px / h_img, 4),
        "was_normalized": was_normalized,
        "thinking_enabled": not no_think,
        "raw_model_output": raw_output,
    }
    
    if thinking:
        result["thinking"] = thinking
    
    # Optionally annotate
    if annotate:
        try:
            annotated_path = annotate_image(image_path, x_px, y_px, annotate_output)
            result["annotated_image"] = annotated_path
        except Exception as e:
            result["annotate_error"] = str(e)
    
    return result


def navigate(
    image_path: str,
    task: str,
    action_history: Optional[List[str]] = None,
    api_url: str = DEFAULT_VLLM_URL,
    model_id: str = DEFAULT_MODEL_ID,
    timeout: int = DEFAULT_TIMEOUT,
    annotate: bool = False,
    annotate_output: Optional[str] = None,
    no_think: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.95,
    api_key: str = DEFAULT_API_KEY,
) -> Dict[str, Any]:
    """
    Get next action using MiMo-VL navigation mode (action-aware, for agents).
    
    Args:
        image_path: Path to the screenshot/image file
        task: Task instruction (e.g., "Search for weather in NYC")
        action_history: List of previous actions taken
        api_url: vLLM API endpoint URL
        model_id: Model identifier
        timeout: API request timeout in seconds
        annotate: Whether to create an annotated image
        annotate_output: Custom path for annotated image
        no_think: Disable reasoning mode
        temperature: Sampling temperature
        top_p: Top-p sampling
        
    Returns:
        Dictionary with action details and coordinates
    """
    # Get image dimensions
    try:
        w_img, h_img = image_size(image_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to read image: {e}"}
    
    # Convert to data URL
    try:
        img_url = image_to_data_url(image_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to encode image: {e}"}
    
    # Call MiMo-VL model
    try:
        raw_output = call_mimo_navigation(
            img_url, task, (w_img, h_img),
            action_history=action_history,
            api_url=api_url, model_id=model_id, timeout=timeout,
            no_think=no_think, temperature=temperature, top_p=top_p, api_key=api_key
        )
    except requests.RequestException as e:
        return {"status": "error", "error": f"API request failed: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error during inference: {e}"}
    
    # Extract thinking
    thinking, clean_output = extract_thinking(raw_output)
    
    # Parse navigation output
    try:
        action_dict = parse_navigation_output(raw_output, (w_img, h_img))
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e),
            "raw_model_output": raw_output,
            "thinking": thinking,
        }
    
    result = {
        "status": "success",
        "task": task,
        "image_path": image_path,
        "image_size": [w_img, h_img],
        "action": action_dict.get("action"),
        "value": action_dict.get("value"),
        "position": action_dict.get("position"),
        "position_px": action_dict.get("position_px"),
        "thinking_enabled": not no_think,
        "raw_model_output": raw_output,
    }
    
    if action_history:
        result["action_history"] = action_history
    
    if thinking:
        result["thinking"] = thinking
    
    # Optionally annotate
    pos_px = action_dict.get("position_px")
    if annotate and pos_px:
        try:
            x_px, y_px = pos_px
            x_px = clamp(x_px, 0, w_img - 1)
            y_px = clamp(y_px, 0, h_img - 1)
            annotated_path = annotate_image(image_path, x_px, y_px, annotate_output)
            result["annotated_image"] = annotated_path
        except Exception as e:
            result["annotate_error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Find UI element coordinates using MiMo-VL-7B-RL vision model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple grounding with reasoning
  python find_coordinates_mimo.py screen.png "search box"
  python find_coordinates_mimo.py screen.png "Log In button" --annotate
  
  # Fast mode without reasoning
  python find_coordinates_mimo.py screen.png "back arrow" --no-think
  
  # Navigation mode for agentic workflows
  python find_coordinates_mimo.py screen.png "Search for weather in NYC" --nav
  python find_coordinates_mimo.py screen.png "Fill the form" --nav --history "['clicked email field']"
  
Environment Variables:
  MIMO_API_URL    - Override default API URL
  MIMO_MODEL_ID   - Override default model ID  
  MIMO_TIMEOUT    - Override default timeout (seconds)

Notes:
  - MiMo-VL uses pixel coordinates by default (not normalized 0-1)
  - Thinking mode provides chain-of-thought reasoning but is slower
  - Use --no-think for faster responses on simple tasks
        """
    )
    
    parser.add_argument("image_path", help="Path to the screenshot/image file")
    parser.add_argument("instruction", help="Element description or task instruction")
    
    # Mode selection
    parser.add_argument("--nav", "-n", action="store_true",
                        help="Use navigation mode (action-aware, for agents)")
    parser.add_argument("--history", metavar="LIST",
                        help="Action history as Python list string (e.g., \"['clicked home']\")")
    
    # Thinking mode
    parser.add_argument("--no-think", action="store_true",
                        help="Disable reasoning mode for faster responses")
    
    # Output options
    parser.add_argument("--compact", action="store_true", help="Compact JSON output (single line)")
    
    # Annotation options
    parser.add_argument("--annotate", "-a", action="store_true", 
                        help="Create an annotated image with the target marked")
    parser.add_argument("--output", "-o", metavar="PATH",
                        help="Output path for annotated image (default: <input>_mimo_annotated.<ext>)")
    
    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    
    # API options
    parser.add_argument("--api-url", default=DEFAULT_VLLM_URL, help="vLLM API endpoint URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key for authentication")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model identifier")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="API request timeout in seconds")
    
    args = parser.parse_args()
    
    # Parse action history if provided
    action_history = None
    if args.history:
        try:
            action_history = ast.literal_eval(args.history)
            if not isinstance(action_history, list):
                action_history = [str(action_history)]
        except (ValueError, SyntaxError):
            action_history = [args.history]
    
    # Run appropriate mode
    if args.nav:
        result = navigate(
            image_path=args.image_path,
            task=args.instruction,
            action_history=action_history,
            api_url=args.api_url,
            model_id=args.model,
            timeout=args.timeout,
            annotate=args.annotate,
            annotate_output=args.output,
            no_think=args.no_think,
            temperature=args.temperature,
            top_p=args.top_p,
            api_key=args.api_key,
        )
    else:
        result = find_coordinates(
            image_path=args.image_path,
            instruction=args.instruction,
            api_url=args.api_url,
            model_id=args.model,
            timeout=args.timeout,
            annotate=args.annotate,
            annotate_output=args.output,
            no_think=args.no_think,
            temperature=args.temperature,
            top_p=args.top_p,
            api_key=args.api_key,
        )
    
    # Output result
    indent = None if args.compact else 2
    print(json.dumps(result, indent=indent))
    
    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()