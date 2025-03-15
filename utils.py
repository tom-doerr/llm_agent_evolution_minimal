def validate_input(input_data: str) -> bool:
    """
    Validate that input is a non-empty string.
    
    Args:
        input_data: The input to validate
        
    Returns:
        True if input is a non-empty string, False otherwise
    """
    return isinstance(input_data, str) and bool(input_data.strip())

def safe_int_conversion(input_str: str) -> int | None:
    """
    Safely convert a string to an integer.
    
    Args:
        input_str: The string to convert to an integer
        
    Returns:
        The integer value if conversion succeeds, None otherwise
    """
    try:
        return int(input_str)
    except (ValueError, TypeError):
        return None
