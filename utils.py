def validate_input(input_data: str) -> bool:
    """
    Validates if the input data is a non-empty string.
    """
    return isinstance(input_data, str) and bool(input_data.strip())

def safe_int_conversion(input_str: str) -> int:
    """
    Safely converts a string to an integer. Returns None if conversion fails.
    """
    try:
        return int(input_str)
    except ValueError:
        return None
