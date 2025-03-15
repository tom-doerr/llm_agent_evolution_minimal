import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

def is_non_empty_string(value: Any) -> bool:
    # Check if value is a non-empty string after stripping whitespace
    return isinstance(value, str) and bool(value.strip())

def safe_int_conversion(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_float_conversion(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def is_valid_number(value: Any) -> bool:
    # Check if value is int/float but not subclass bool
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def truncate_string(value: Any, max_length: int = 100) -> str:
    if not isinstance(value, str):  # Handle non-string inputs gracefully
        return ""
    if len(value) <= max_length:
        return value
    return value[:max_length] + "..."

def run_inference(input_string: str, model: str = "deepseek/deepseek-reasoner") -> str:
    """Run inference using the specified model."""
    try:
        import litellm
        
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": input_string}],
            stream=False
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        return f"Inference result for: {input_string}"
    except ImportError:
        # Fallback if litellm is not installed
        return f"Inference result for: {input_string} (litellm not installed)"
    except Exception as e:
        return f"Error during inference: {str(e)}"

def extract_xml(xml_string: str) -> str:
    """Extract XML data from a string."""
    if not is_non_empty_string(xml_string):
        return ""
        
    # Try to find XML content between tags if it's embedded in other text
    try:
        # Look for content between first < and last >
        start_idx = xml_string.find('<')
        end_idx = xml_string.rfind('>')
        
        if start_idx >= 0 and end_idx > start_idx:
            xml_content = xml_string[start_idx:end_idx+1]
            root = ET.fromstring(xml_content)
            return ET.tostring(root, encoding='unicode')
        else:
            # Try parsing the whole string as XML
            root = ET.fromstring(xml_string)
            return ET.tostring(root, encoding='unicode')
    except ET.ParseError:
        return ""

def parse_xml_to_dict(xml_string: str) -> Dict[str, Any]:
    """Parse XML string into a dictionary structure."""
    if not is_non_empty_string(xml_string):
        return {}
    
    try:
        root = ET.fromstring(xml_string)
        result = {}
        
        for child in root:
            result[child.tag] = child.text
            
        return result
    except ET.ParseError:
        return {}
