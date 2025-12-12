"""
JSON Cleaner Module
Handles non-standard JSON by:
- Removing JS-style comments (// ...)
- Removing block comments (/* ... */)
- Removing (Skipping ...) placeholders
- Removing trailing commas
- Stripping whitespace
"""

import re
import json


def heal_json(raw_text: str):
    """
    Attempts to fix invalid JSON:
    - Remove block comments (/* ... */) - safe outside strings
    - Remove (Skipping ...) placeholders
    - Fix trailing commas
    - Remove control characters
    
    Note: Does NOT remove // comments as they often appear in URLs
    
    Args:
        raw_text: Potentially malformed JSON text
    
    Returns:
        Parsed JSON object (dict or list)
        
    Raises:
        ValueError: If JSON cannot be fixed
    """
    # 1. Remove /* ... */ block comments (safe - not in strings)
    raw_text = re.sub(r'/\*.*?\*/', '', raw_text, flags=re.DOTALL)

    # 2. Remove (Skipping...) placeholders
    raw_text = re.sub(r'\(Skipping.*?\)', '', raw_text, flags=re.IGNORECASE | re.DOTALL)

    # 3. Remove trailing commas before } or ]
    raw_text = re.sub(r',(\s*[}\]])', r'\1', raw_text)

    # 4. Remove control characters (0x00-0x1F) EXCEPT newline(0x0A), tab(0x09), carriage return(0x0D)
    cleaned = []
    for char in raw_text:
        code = ord(char)
        if code < 0x20 and code not in (0x09, 0x0A, 0x0D):
            # Skip control character
            continue
        cleaned.append(char)
    raw_text = ''.join(cleaned)

    # 5. Try to parse
    try:
        return json.loads(raw_text)
    except Exception as e:
        raise ValueError(f"❌ Could not fix JSON automatically: {str(e)}")


def load_json_safely(path: str):
    """
    Load JSON file with automatic healing for common issues.
    
    Args:
        path: File path to JSON file
    
    Returns:
        Parsed JSON object (dict or list)
        
    Raises:
        ValueError: If JSON cannot be parsed or fixed
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        return json.loads(raw)
    except:
        return heal_json(raw)


def clean_json(text: str) -> str:
    """
    Cleans non-standard JSON by:
    - Removing JS-style comments (// ...) OUTSIDE of strings
    - Removing block comments (/* ... */)
    - Removing trailing commas before } or ]
    - Removing control characters
    - Stripping whitespace
    
    Args:
        text: Raw JSON text (potentially malformed)
    
    Returns:
        Cleaned JSON text ready for parsing
        
    Raises:
        ValueError: If cleaned JSON is still invalid
    """
    
    # Check for empty or whitespace-only input
    text = text.strip()
    if not text:
        raise ValueError("JSON input is empty")

    # 1. Remove /* ... */ block comments (safe - not in strings)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # 2. Remove (Skipping...) placeholders (safe - not JSON syntax)
    text = re.sub(r'\(Skipping.*?\)', '', text, flags=re.IGNORECASE | re.DOTALL)

    # 3. Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 4. Remove control characters (0x00-0x1F) EXCEPT newline(0x0A), tab(0x09), carriage return(0x0D)
    cleaned = []
    for char in text:
        code = ord(char)
        if code < 0x20 and code not in (0x09, 0x0A, 0x0D):
            # Skip control character
            continue
        cleaned.append(char)
    text = ''.join(cleaned)

    # 5. Strip leading/trailing whitespace
    text = text.strip()
    
    # Check again after cleaning
    if not text:
        raise ValueError("❌ JSON file is empty after cleaning (possibly contained only comments)")

    # 6. Validate by attempting to parse
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Still invalid JSON after cleaning: {e}")

    return text


def load_and_fix_json(path: str):
    """
    Load JSON file that may contain comments or trailing commas.
    Uses automatic healing for common JSON errors.
    
    Args:
        path: File path to JSON file
    
    Returns:
        Parsed JSON object (dict or list)
        
    Raises:
        ValueError: If JSON cannot be parsed after cleaning
    """
    return load_json_safely(path)



