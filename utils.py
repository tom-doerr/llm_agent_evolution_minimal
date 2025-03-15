from typing import Any

def is_non_empty_string(value: str) -> bool:
    # Check if value is a non-empty string after stripping whitespace
    return isinstance(value, str) and bool(value.strip())

def safe_int_conversion(value: str) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_float_conversion(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def is_valid_number(value: Any) -> bool:
    # Check if value is int/float but not subclass bool
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def truncate_string(value: str, max_length: int = 100) -> str:
    if not isinstance(value, str):  # Handle non-string inputs gracefully
        return ""
    if len(value) <= max_length:
        return value
    return value[:max_length] + "..."

def run_inference(input_string: str) -> str:
    """Placeholder function for running inference."""
    return f"Inference result for: {input_string}"

def extract_xml(input_string: str) -> str:
    """Placeholder function for extracting XML data."""
    return f"Extracted XML data from: {input_string}"
