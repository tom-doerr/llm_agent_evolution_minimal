def validate_input(input_data: str) -> bool:
    # Check if input is a non-empty string
    return isinstance(input_data, str) and bool(input_data.strip())

def safe_int_conversion(input_str: str) -> int | None:
    # Convert string to int, return None if conversion fails
    try:
        return int(input_str)
    except (ValueError, TypeError):
        return None
