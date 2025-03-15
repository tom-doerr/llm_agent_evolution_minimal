def validate_input(input_data: str) -> bool:
    return isinstance(input_data, str) and bool(input_data.strip())

def safe_int_conversion(input_str: str) -> int | None:
    try:
        return int(input_str)
    except ValueError:
        return None
