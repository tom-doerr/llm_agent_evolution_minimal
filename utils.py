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

def run_inference(input_string: str, model: str = "deepseek/deepseek-reasoner", stream: bool = False) -> str:
    try:
        import litellm
        import os
        
        # Ensure API key is set for DeepSeek models
        if model.startswith("deepseek/") and "DEEPSEEK_API_KEY" not in os.environ:
            return f"Error: DEEPSEEK_API_KEY environment variable not set for model {model}"
            
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": input_string}],
            stream=stream
        )
        
        # Handle streaming response
        if stream:
            full_response = ""
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content
                    elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        full_response += delta.reasoning_content
            return full_response
        
        # Handle non-streaming response
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            elif hasattr(choice, 'text'):
                return choice.text
        
        return ""
    except ImportError:
        # Fallback if litellm is not installed
        return f"Error: litellm not installed"
    except Exception as e:
        return f"Error during inference: {str(e)}"

def extract_xml(xml_string: str) -> str:
    """Extract valid XML content from a string that might contain other text"""
    if not is_non_empty_string(xml_string):
        return ""
    
    # Try to find XML content between tags if it's embedded in other text
    try:
        # Look for content between first < and last >
        start_idx = xml_string.find('<')
        end_idx = xml_string.rfind('>')
        
        if start_idx >= 0 and end_idx > start_idx:
            xml_content = xml_string[start_idx:end_idx+1]
            try:
                root = ET.fromstring(xml_content)
                return ET.tostring(root, encoding='unicode')
            except ET.ParseError:
                # Try to find a complete XML element
                for tag in ['response', 'result', 'data', 'xml', 'root']:
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
        
        # Try parsing the whole string as XML
        root = ET.fromstring(xml_string)
        return ET.tostring(root, encoding='unicode')
    except ET.ParseError:
        return ""

def parse_xml_to_dict(xml_string: str) -> Dict[str, Any]:
    if not is_non_empty_string(xml_string):
        return {}
    
    try:
        # First try to extract valid XML if it's embedded in other text
        extracted = extract_xml(xml_string)
        if extracted:
            xml_string = extracted
            
        root = ET.fromstring(xml_string)
        result = {}
        
        for child in root:
            # Handle nested elements
            if len(child) > 0:
                result[child.tag] = parse_xml_element(child)
            else:
                result[child.tag] = child.text
            
        return result
    except ET.ParseError:
        return {}

def parse_xml_element(element: ET.Element) -> Union[Dict[str, Any], str]:
    if len(element) == 0:
        return element.text or ""
    
    result = {}
    for child in element:
        if len(child) > 0:
            result[child.tag] = parse_xml_element(child)
        else:
            result[child.tag] = child.text or ""
    
    return result

class Agent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.memory = []
        self.lm = None
    
    def __str__(self) -> str:
        return f"Agent with model: {self.model_name}"
    
    def __call__(self, input_text: str) -> str:
        """Process input and return response"""
        result = self.run(input_text)
        self.memory.append({"input": input_text, "output": result})
        return result
    
    def run(self, input_text: str) -> str:
        """Run inference using the agent's language model or fallback to run_inference"""
        if self.lm and hasattr(self.lm, 'complete'):
            try:
                response = self.lm.complete(input_text)
                if hasattr(response, 'completion'):
                    return response.completion
                return str(response)
            except Exception as e:
                return f"Error using DSPy LM: {str(e)}"
        
        # Fallback to run_inference
        # Use streaming for DeepSeek models to properly handle reasoning content
        use_stream = self.model_name.startswith("deepseek/")
        return run_inference(input_text, self.model_name, stream=use_stream)

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
        # Initialize the language model
        agent.lm = dspy.LM(model_name)
        return agent
    except ImportError:
        # Return the agent without DSPy integration
        return agent
    except Exception as e:
        # Return agent with error info in model_name
        agent.model_name = f"{model_name} (Error: {str(e)})"
        return agent
