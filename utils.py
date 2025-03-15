import xml.etree.ElementTree as ET
import datetime
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

def run_inference(input_string: str, model: str = "deepseek/deepseek-reasoner", stream: bool = False) -> str:
    try:
        import litellm
        import os
        
        if model.startswith("deepseek/") and "DEEPSEEK_API_KEY" not in os.environ:
            return f"Error: DEEPSEEK_API_KEY environment variable not set for model {model}"
            
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": input_string}],
            stream=stream
        )
        
        if stream:
            full_response = ""
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content
                    elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        full_response += delta.reasoning_content
            return full_response
        
        if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            elif hasattr(choice, 'text'):
                return choice.text
        
        return ""
    except ImportError:
        return "Error: litellm not installed"
    except Exception as e:
        return f"Error during inference: {str(e)}"

def extract_xml(xml_string: str) -> str:
    """Extract valid XML content from a string that might contain other text"""
    if not is_non_empty_string(xml_string):
        return ""
    
    try:
        # First try to parse the entire string as XML
        try:
            root = ET.fromstring(xml_string)
            return ET.tostring(root, encoding='unicode')
        except ET.ParseError:
            pass
            
        # Try to extract XML content between tags
        start_idx = xml_string.find('<')
        end_idx = xml_string.rfind('>')
        
        if start_idx >= 0 and end_idx > start_idx:
            xml_content = xml_string[start_idx:end_idx+1]
            try:
                root = ET.fromstring(xml_content)
                return ET.tostring(root, encoding='unicode')
            except ET.ParseError:
                # Try common XML root tags
                for tag in ['response', 'result', 'data', 'xml', 'root', 'memory', 'action']:
                    start_tag = f'<{tag}'
                    end_tag = f'</{tag}>'
                    start = xml_string.find(start_tag)
                    end = xml_string.find(end_tag)
                    if start >= 0 and end > start:
                        try:
                            element = xml_string[start:end + len(end_tag)]
                            root = ET.fromstring(element)
                            return ET.tostring(root, encoding='unicode')
                        except ET.ParseError:
                            continue
        return ""
    except ET.ParseError:
        return ""

def parse_xml_to_dict(xml_string: str) -> Dict[str, Any]:
    if not is_non_empty_string(xml_string):
        return {}
    
    try:
        extracted = extract_xml(xml_string)
        if not extracted:
            return {}
            
        xml_string = extracted
        root = ET.fromstring(xml_string)
        result = {}
        
        # Include root attributes if any
        if root.attrib:
            result.update(root.attrib)
        
        # Process child elements
        for child in root:
            if len(child) > 0:
                result[child.tag] = parse_xml_element(child)
            else:
                result[child.tag] = child.text or ""
            
        return result
    except ET.ParseError:
        return {}

def parse_xml_element(element: ET.Element) -> Union[Dict[str, Any], str]:
    if len(element) == 0:
        # Return text with attributes if any
        if element.attrib:
            result = {"_text": element.text or ""}
            result.update(element.attrib)
            return result
        return element.text or ""
    
    result = {}
    # Include element attributes if any
    if element.attrib:
        result.update(element.attrib)
    
    # Process child elements
    for child in element:
        if child.tag in result:
            # Handle duplicate tags by converting to list
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(parse_xml_element(child))
        else:
            result[child.tag] = parse_xml_element(child)
    
    return result

def print_datetime() -> None:
    """Print the current date and time."""
    current_time = datetime.datetime.now()
    print(f"Current date and time: {current_time}")

class Agent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.memory: List[Dict[str, str]] = []
        self.lm = None
    
    def __str__(self) -> str:
        return f"Agent with model: {self.model_name}"
    
    def __repr__(self) -> str:
        return f"Agent(model_name='{self.model_name}', memory_size={len(self.memory)})"
    
    def __call__(self, input_text: str) -> str:
        result = self.run(input_text)
        self.memory.append({"input": input_text, "output": result})
        return result
    
    def run(self, input_text: str) -> str:
        if not is_non_empty_string(input_text):
            return ""
            
        if self.lm and hasattr(self.lm, 'complete'):
            try:
                response = self.lm.complete(input_text)
                if hasattr(response, 'completion'):
                    return response.completion
                return str(response)
            except Exception as e:
                return f"Error using DSPy LM: {str(e)}"
        
        use_stream = self.model_name.startswith("deepseek/")
        return run_inference(input_text, self.model_name, stream=use_stream)
    
    def clear_memory(self) -> None:
        """Clear the agent's memory"""
        self.memory = []

def create_agent(model_type: str) -> Agent:
    model_mapping = {
        'flash': 'openrouter/google/gemini-2.0-flash-001',
        'pro': 'openrouter/google/gemini-2.0-pro-001',
        'deepseek': 'deepseek/deepseek-reasoner',
        'default': 'openrouter/google/gemini-2.0-flash-001'
    }
    
    model_name = model_mapping.get(model_type.lower(), model_mapping['default'])
    agent = Agent(model_name)
    
    try:
        import dspy
        agent.lm = dspy.LM(model_name)
    except ImportError:
        pass
    except Exception as e:
        agent.model_name = f"{model_name} (Error: {str(e)})"
    
    return agent
