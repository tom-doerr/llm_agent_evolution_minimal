def validate_input(input_data: str) -> bool:
    """Check if input is a non-empty string."""
    return isinstance(input_data, str) and bool(input_data.strip())

def safe_int_conversion(input_str: str) -> int | None:
    """Convert string to int, return None if conversion fails."""
    try:
        return int(input_str)
    except (ValueError, TypeError):
        return None

def safe_float_conversion(input_str: str) -> float | None:
    """Convert string to float, return None if conversion fails."""
    try:
        return float(input_str)
    except (ValueError, TypeError):
        return None

def is_valid_number(value) -> bool:
    """Check if a value is a valid number (int or float)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to specified length and add ellipsis if needed."""
    if not isinstance(text, str):
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
