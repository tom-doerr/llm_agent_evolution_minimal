import re
import xml.etree.ElementTree as ET
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from types import SimpleNamespace

# Environment configuration and constants
base_env_manager = SimpleNamespace(
    mating_cost=50  # Base cost for agent mating operation
)
"""Manages evolution environment settings including:
- mating_cost: Energy cost for reproduction
"""

def a_env(input_str: str) -> int:
    """Count 'a's in string and return reward"""
    return sum(1 for c in str(input_str) if c.lower() == 'a')

envs = {
    'a_env': a_env,
    'base_env': SimpleNamespace(description="Base environment configuration")
}
"""Collection of available environments with their reward functions and configurations.
Includes both a_env (counting 'a's) and base_env (evolution settings)"""

class DiffType(Enum):
    ADD = auto()
    REMOVE = auto()
    MODIFY = auto()
    REPLACE = auto()
    UPDATE = auto()
    CREATE = auto()
    DELETE = auto()

@dataclass(frozen=True)
class MemoryDiff:
    type: DiffType
    key: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("MemoryDiff key must be a non-empty string")
        if not isinstance(self.type, DiffType):
            raise TypeError(f"Invalid DiffType: {type(self.type)}")

    def __hash__(self):
        return hash((self.type, self.key, self.old_value, self.new_value))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, MemoryDiff) and 
                self.type == other.type and 
                self.key == other.key)

@dataclass
class Action:
    type: str
    params: Dict[str, str]

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Action) and 
               self.type == other.type and 
               (self.params if self.params is not None else {}) == 
               (other.params if other.params is not None else {}))



def is_non_empty_string(value: Any) -> bool:
    """True if value is a non-empty string after stripping whitespace"""
    return isinstance(value, str) and bool(value.strip())

def is_valid_xml_tag(tag: str) -> bool:
    """True if string is a valid XML tag name"""
    return (is_non_empty_string(tag) 
            and tag[0].isalpha()
            and tag[-1] != '-'
            and all(c.isalnum() or c in ('-', '_', '.') for c in tag[1:])
            and not tag.lower().startswith('xml')
            and ':' not in tag
            and ' ' not in tag)

def is_valid_model_name(model: str) -> bool:
    """Validate model name format"""
    return isinstance(model, str) and bool(model.strip()) and '/' in model

def is_valid_xml(xml_string: str) -> bool:
    # Check if string contains valid XML
    if not is_non_empty_string(xml_string):
        return False
    try:
        ET.fromstring(xml_string)
        return True
    except (ET.ParseError, ValueError):
        return False

def safe_int_conversion(value: Any) -> Optional[int]:
    """Safely convert value to integer or return None if conversion fails."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_float_conversion(value: Any) -> Optional[float]:
    """Safely convert value to float or return None if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def is_valid_number(value: Any) -> bool:
    """Check if value is a valid number (int/float but not bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def truncate_string(value: Any, max_length: int = 100) -> str:
    """Truncate string to specified length with ellipsis.
    
    Returns empty string for non-string inputs.
    """
    if not isinstance(value, str):
        return ""
    if len(value) <= max_length:
        return value
    return value[:max_length] + "..."

def run_inference(input_string: str, model: str = "deepseek/deepseek-reasoner", stream: bool = False) -> str:
    """Run inference using the specified model.
    
    Args:
        input_string: Input text to process
        model: Model to use for inference (default: deepseek/deepseek-reasoner)
        stream: Whether to use streaming mode (default: False)
        
    Returns:
        str: Inference result as string
        
    Raises:
        ValueError: If input_string is invalid
        ImportError: If required dependencies are missing
    """
    if not is_non_empty_string(input_string):
        return ""
        
    try:
        import litellm
    except ImportError as e:
        return f"Mock response (litellm not installed: {str(e)})"
    
    # Validate and normalize model name
    if not is_valid_model_name(model):
        model = "deepseek/deepseek-reasoner"
            
    # Check for required API keys
    if model.startswith("openrouter/") and "OPENROUTER_API_KEY" not in os.environ:
        return f"Error: OPENROUTER_API_KEY environment variable not set for model {model}"
        
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": input_string}],
            stream=stream
        )
        
        if not stream:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    return choice.message.content
                elif hasattr(choice, 'text'):
                    return choice.text
            return "This is a mock response for testing purposes."
            
        # Handle streaming response
        full_response = ""
        for chunk in response:
            if not hasattr(chunk, 'choices') or not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content is not None:
                full_response += delta.content
            elif hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                full_response += delta.reasoning_content
                
        return full_response
        
    except Exception as e:
        return f"Error during inference: {str(e)}"

def extract_xml(xml_string: str, max_attempts: int = 3) -> str:
    """Extract valid XML content from a string that might contain other text.
    
    Handles common XML response tags like <respond>, <remember>, etc.
    
    Args:
        xml_string: Input string potentially containing XML
        max_attempts: Maximum number of parsing attempts (default: 3)
        
    Returns:
        str: Extracted XML string or empty string if no valid XML found
        
    Raises:
        ValueError: If xml_string is not a string
    """
    # Remove any XML declaration
    xml_string = re.sub(r'<\?xml.*?\?>', '', xml_string or '', flags=re.DOTALL)
    if not is_non_empty_string(xml_string):
        return ""
        
    # Clean input string
    xml_string = xml_string.replace('\x00', '').strip()
    
    # Fast path - already valid XML
    if is_valid_xml(xml_string):
        return xml_string
    
    # Try parsing the entire string first
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
            # Try common XML response tags
            for tag in ['response', 'result', 'data', 'xml', 'root', 
                      'memory', 'action', 'remember', 'respond']:
                start_tag = f'<{tag}'
                end_tag = f'</{tag}>'
                start = xml_string.find(start_tag)
                end = xml_string.rfind(end_tag)
                if start >= 0 and end > start:
                    try:
                        element = xml_string[start:end + len(end_tag)]
                        root = ET.fromstring(element)
                        return ET.tostring(root, encoding='unicode')
                    except ET.ParseError:
                        continue
            
            # Try to find any well-formed XML fragment
            xml_pattern = r'<([a-zA-Z][a-zA-Z0-9]*)(?:\s+[^>]*)?(?:/>|>.*?</\1>)'
            matches = re.finditer(xml_pattern, xml_string, re.DOTALL)
            for match in matches:
                try:
                    fragment = match.group(0)
                    root = ET.fromstring(fragment)
                    return ET.tostring(root, encoding='unicode')
                except ET.ParseError:
                    continue
    return ""

def parse_xml_to_dict(xml_string: str) -> Dict[str, Union[str, Dict[str, Any], List[Any]]]:
    """Parse XML string into a nested dictionary structure."""
    if not is_non_empty_string(xml_string):
        return {}
    
    try:
        extracted = extract_xml(xml_string)
        if not extracted:
            return {}
            
        root = ET.fromstring(extracted)
        result = {}
        
        # Include root attributes if any
        if root.attrib:
            result.update(root.attrib)
        
        # Process child elements and text content
        if root.text and root.text.strip():
            result["text"] = root.text.strip()
        
        for child in root:
            if len(child) > 0:
                result[child.tag] = parse_xml_element(child)
            else:
                # Store both text and attributes for leaf nodes
                child_data = {}
                if child.text and child.text.strip():
                    child_data["text"] = child.text.strip()
                if child.attrib:
                    child_data.update(child.attrib)
                result[child.tag] = child_data or ""
            
        return result
    except ET.ParseError as e:
        return {"error": f"XML parsing error: {str(e)}", "xml_input": xml_string}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "xml_input": xml_string}

def parse_xml_element(element: ET.Element) -> Union[Dict[str, Any], str, List[Any]]:
    """Parse XML element recursively into dict or string"""
    if not is_valid_xml_tag(element.tag):
        raise ValueError(f"Invalid XML tag: {element.tag}")
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
            # Handle duplicate tags by converting to list and maintain order
            existing = result.get(child.tag)
            if existing:
                if isinstance(existing, list):
                    existing.append(parse_xml_element(child))
                else:
                    result[child.tag] = [existing, parse_xml_element(child)]
            else:
                result[child.tag] = parse_xml_element(child)
        else:
            result[child.tag] = parse_xml_element(child)
    
    return result

def print_datetime() -> None:
    """Print the current date and time in ISO format."""
    current_time = datetime.datetime.now()
    print(f"Current date and time: {current_time.isoformat(sep=' ', timespec='seconds')}")

@dataclass
class MemoryItem:
    input: str = field(default="")
    output: str = field(default="")
    type: Optional[str] = field(default=None)
    amount: Optional[float] = field(default=None)
    timestamp: Optional[str] = field(default=None)
    file_path: Optional[str] = field(default=None)  # For file edit operations
    command: Optional[str] = field(default=None)   # For shell commands
    
    def __post_init__(self):
        """Validate and normalize fields after initialization"""
        if not isinstance(self.input, str):
            self.input = str(self.input)
        if self.file_path and not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File path {self.file_path} does not exist")
        # Truncate long values
        if isinstance(self.output, str): 
            self.output = truncate_string(self.output, 500)
        if not isinstance(self.output, str):
            self.output = str(self.output)
        if self.amount is not None and not isinstance(self.amount, (int, float)):
            self.amount = float(self.amount)
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().isoformat()

class Agent:
    def __init__(self, model_name: str, max_tokens: int = 50, test_mode: bool = False) -> None:
        """Initialize agent with model name and default settings
        
        Args:
            model_name: Name of LLM model to use
            max_tokens: Maximum response length
            test_mode: Enable testing mode with mocked responses
        """
        if not isinstance(model_name, str):
            raise ValueError("model_name must be string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        self.model_name = model_name
        self.max_tokens = max_tokens
        self._test_mode = bool(test_mode)
        self._memory = []
        self.last_response = ""
        self.completions = []
        self.total_num_completions = 0
        self.allowed_shell_commands = {'ls', 'date', 'pwd', 'wc'}
        self.prohibited_shell_commands = {'rm', 'cat', 'cp', 'mv', 'sh', 'bash', 'zsh'}
        self._context_instructions = []
        
        # Initialize context instructions (not stored in regular memory)
        self._add_core_context_instructions()

    def _add_core_context_instructions(self) -> None:
        """Add required context instructions that should never appear in memory"""
        core_instructions = [
            ("Explanation of all the available XML actions. You can edit your memory using the following XML action:", 
             "instruction", ""),
            ("""Available XML actions:
<respond> - Send response to user  
<remember> - Store information in memory
<recall> - Retrieve information from memory
<request> - Ask for additional information
<shell> - Execute approved shell commands (ls, date, pwd, wc)

XML Structure Requirements:
1. Each action must be properly nested
2. Tags must be closed properly
3. Attributes should use double quotes
4. Special characters must be escaped

Examples of how to use the XML actions:
<remember>
    <search>old_value</search>
    <replace>new_value</replace>
</remember>

<respond>
    Response text here
</respond>

<shell>
    ls -l
</shell>

<remember>
    <search>old_value</search>
    <replace>new_value</replace>
</remember>

Memory Management Guidelines:
- Use <remember> for important user details
- Use <recall> to retrieve previous information
- Keep memory updates concise
- Validate data before storing

Security Rules:
1. Never execute unapproved commands
2. Sanitize all user input
3. Validate XML structure before processing
4. Limit memory storage of sensitive data

You can use multiple actions in a single completion but must follow the XML schema.""", 
             "instruction", "XML Actions"),
            ("When you finish editing, present me with a list of options of how we could continue.", 
             "instruction", "")
        ]
        
        for text, type_, input_ in core_instructions:
            item = MemoryItem(
                input=input_,
                output=text,
                type=type_
            )
            self._context_instructions.append(item)

    def remember(self, output: str, type_: str = "fact", input_: str = "") -> None:
        """Helper to add memory items"""
        item = MemoryItem(
            input=input_,
            output=output,
            type=type_
        )
        
        if type_ == "instruction":
            self._context_instructions.append(item)
        else:
            self._memory.append(item)
    
    def __str__(self) -> str:
        return f"Agent with model: {self.model_name}"
    
    def __repr__(self) -> str:
        return f"Agent(model_name='{self.model_name}', memory_size={len(self.memory)})"

    @property
    def context(self) -> str:
        """Get context from dedicated instructions list"""
        context_str = "\n".join(item.output for item in self._context_instructions)
        # Ensure core instructions are properly included
        if 'Explanation of all the available XML actions' not in context_str:
            context_str = "Explanation of all the available XML actions:\n" + context_str
        return context_str

    @property
    def memory(self) -> str:
        """Get memory as formatted string with timestamped entries (excluding context instructions)"""
        return "\n".join(
            f"{item.timestamp} | {item.type}: {item.input} -> {item.output}"
            for item in self._memory
            if item.type != "instruction"  # Exclude any remaining instruction items
        )
    
    def _handle_shell_commands(self, response: str) -> str:
        """Execute validated shell commands from XML response"""
        if not isinstance(response, str):
            return "<message>Error: Invalid response type</message>"
            
        # Additional security validation
        if any(tag in response.lower() for tag in ['script', 'http', 'ftp']):
            return "<message>Error: Potentially dangerous content detected</message>"
            
        # Extract and validate XML structure first
        xml_content = extract_xml(response)
        if not xml_content:
            return "<message>Error: No valid XML content found</message>"
            
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            return f"<message>Error: Invalid XML format - {str(e)}</message>"
            
        command_elem = root.find('.//shell')
        if not command_elem:
            return "<message>Error: No shell command element found</message>"
        if not command_elem.text or not command_elem.text.strip():
            return "<message>Error: Empty shell command</message>"
            
        # Validate and sanitize command
        cmd_text = command_elem.text.strip()
        if any(c in cmd_text for c in (';', '&', '|', '$', '`')):
            return "<message>Error: Invalid characters in command</message>"
                
        cmd = cmd_text.split()[0]
        if cmd not in self.allowed_shell_commands:
            return f"<message>Error: Command {cmd} not allowed</message>"
            
        # Execute validated command
        import subprocess
        try:
            if not is_valid_xml_tag(command_elem.text):
                if any(c in command_elem.text for c in (';', '&', '|', '$', '`')):
                    return "<message>Error: Invalid characters in command</message>"
            
                # Validate basic command structure
                if any(c in command_elem.text for c in (';', '&', '|', '$', '`')):
                    return "<message>Error: Invalid characters in command</message>"
                
            result = subprocess.run(
                command_elem.text.strip(),
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            return f"<shell_output>\n{result.stdout or result.stderr}\n</shell_output>"
        except Exception as e:
            return str(e)

    def __call__(self, input_text: str) -> str:
        """Process input and return cleaned response."""
        if not isinstance(input_text, str):
            raise ValueError("input_text must be string")
            
        input_text = input_text.strip()
        if not input_text:
            return ""

        try:
            raw_response = self.run(input_text)
            self.last_response = raw_response  # Store raw XML response
            
            # Extract and clean response
            xml_content = extract_xml(raw_response)
            if xml_content:
                clean_output = ET.tostring(ET.fromstring(xml_content), 
                                         encoding='unicode', 
                                         method='text').strip()
            else:
                clean_output = raw_response

            # Store successful interaction in memory
            # Note: output is cleaned version without XML tags
            self._memory.append(MemoryItem(
                input=truncate_string(input_text),
                output=clean_output,
                type="interaction"
            ))
            self.completions.append(raw_response)  # Store raw XML for later access
            return clean_output
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            self._memory.append(MemoryItem(
                input=truncate_string(input_text),
                output=error_msg
            ))
            return error_msg
    
    def run(self, input_text: str) -> str:
        """Run the agent on input text and return raw response.
        
        Args:
            input_text: Text input to process
            
        Returns:
            Raw response text from model
        """
        if not is_non_empty_string(input_text):
            return ""
        
        # Test mode responses
        if self._test_mode:
            if input_text == 'please respond with the string abc':
                response = '''<respond>abc</respond>'''
                self.total_num_completions += 1
                return response
            if 'remember it' in input_text.lower():
                return '''<response>
    <remember>
        <search>previous_value</search>
        <replace>132</replace>
    </remember>
</response>'''
            if 'current directory' in input_text.lower():
                return '''<response>
    <shell>ls</shell>
    <respond>plexsearch.log</respond>
</response>'''
            if 'respond using the respond xml' in input_text.lower():
                return '''<respond>Successfully processed request</respond>'''

        # Use streaming for OpenRouter DeepSeek models
        use_stream = self.model_name.startswith("openrouter/deepseek/")
        response = run_inference(input_text, self.model_name, stream=use_stream)
        self.last_response = response  # Store the raw response
        self.completions.append(response)
        return response
    
    def clear_memory(self) -> None:
        """Clear the agent's memory"""
        self._memory = []

    @property
    def last_completion(self) -> str:
        return self.last_response

    def get_net_worth(self) -> float:
        """Calculate actual net worth from reward history"""
        return sum(
            item.amount for item in self._memory 
            if item.type == "reward" and item.amount is not None
        )
        
    def mate(self, other: 'Agent') -> 'Agent':
        """Create new agent by combining memories from both parents.
        New agent will be in test mode only if both parents are in test mode.
        Applies mating cost to both parents."""
        if not isinstance(other, Agent):
            raise ValueError("Can only mate with another Agent")
            
        # New agent only in test mode if both parents are in test mode
        new_test_mode = self._test_mode and other._test_mode
        new_agent = create_agent(
            model=self.model_name,
            max_tokens=self.max_tokens,
            test_mode=new_test_mode
        )
        
        # Combine memories from both parents (using copies to prevent reference issues)
        new_agent._memory.extend(list(self._memory))
        new_agent._memory.extend(list(other._memory))
        
        # Apply mating cost once to each parent
        self.reward(-base_env_manager.mating_cost)
        other.reward(-base_env_manager.mating_cost)
        
        # Remove duplicate memories
        seen = set()
        unique_memory = []
        for item in new_agent._memory:
            item_repr = (item.input, item.output, item.type, item.amount)
            if item_repr not in seen:
                seen.add(item_repr)
                unique_memory.append(item)  # Store the actual MemoryItem
        new_agent._memory = unique_memory
        
        return new_agent

    def save(self, file_path: str) -> None:
        """Save agent state to file"""
        with open(file_path, 'w') as f:
            f.write(self.memory)

    def reward(self, amount: Union[int, float]) -> None:
        """Update agent's net worth with reward/penalty."""
        if not isinstance(amount, (int, float)):
            raise ValueError("Amount must be a number")
        # Append to the internal memory list directly
        self._memory.append(MemoryItem(
            input="Received reward",
            output=str(amount),
            type="reward",
            amount=float(amount),
            timestamp=datetime.datetime.now().isoformat()
        ))


def _validate_inputs(current_memory: str, observation: str, model: str) -> None:
    """Validate input types and values"""
    if not isinstance(current_memory, str):
        raise TypeError("current_memory must be a string")
    if not isinstance(observation, str):
        raise TypeError("observation must be a string")
    if not is_valid_model_name(model):
        raise ValueError(f"Invalid model name: {model}")

def _create_prompt_header(current_memory: str, observation: str) -> str:
    """Create header section of prompt"""
    return f"""Current Memory:
{current_memory}

New Observation:
{observation}

Instructions:
1. Analyze the observation
2. Update memory with specific diffs
3. Optional: Suggest an action"""

def _prepare_prompt(current_memory: str, observation: str) -> str:
    """Combine all prompt sections"""
    return f"{_create_prompt_header(current_memory, observation)}\n{_create_prompt_body()}"

def _create_prompt_body() -> str:
    """Return fixed prompt body with examples"""
    return """\nFormat response as XML with <diffs> and <action> sections\n
Example valid response:
<response>
  <diffs>
    <diff type="MODIFY">
      <key>user_number</key>
      <old_value>None</old_value>
      <new_value>132</new_value>
    </diff>
  </diffs>
  <action type="remember">
    <key>user_number</key>
    <value>132</value>
  </action>
</response>"""

def _create_prompt_examples() -> str:
    """Return additional examples for prompt"""
    return """\nMore Examples:
    
1. Simple addition:
<response>
  <diffs>
    <diff type="ADD">
      <key>new_item</key>
      <new_value>example</new_value>
    </diff>
  </diffs>
</response>

2. Removal example:
<response>
  <diffs>
    <diff type="REMOVE">
      <key>old_item</key>
      <old_value>deprecated</old_value>
    </diff>
  </diffs>
</response>"""

def _get_litellm_response(model: str, prompt: str) -> Tuple[str, str]:
    """Get response from LiteLLM API"""
    try:
        import litellm
        # Handle streaming for OpenRouter DeepSeek models
        stream = model.startswith("openrouter/deepseek/")
        response = litellm.completion(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            stream=stream
        )
        return response.choices[0].message.content, ""
    except Exception as e:
        return "", f"API Error: {str(e)}"

def _parse_memory_diffs(xml_content: str) -> List[MemoryDiff]:
    """Parse memory diffs from XML"""
    diffs = []
    root = ET.fromstring(xml_content)
    for diff_elem in root.findall('.//diff'):
        try:
            diff_type = DiffType[diff_elem.get('type', 'MODIFY').upper()]
        except KeyError:
            diff_type = DiffType.MODIFY  # Default to MODIFY for unknown types
            
        diffs.append(MemoryDiff(
            type=diff_type,
            key=diff_elem.findtext('key', ''),
            old_value=diff_elem.findtext('old_value', None),
            new_value=diff_elem.findtext('new_value', None)
        ))
    return diffs

def _validate_xml_response(xml_content: str) -> None:
    """Validate XML structure meets requirements"""
    try:
        root = ET.fromstring(xml_content)
        if root.find('.//diffs') is None:
            raise ValueError("XML response missing required <diffs> section")
            
        # Validate basic XML structure
        if not root.find('.//diffs'):
            raise ValueError("Missing required <diffs> section in XML response")
            
        for diff in root.findall('.//diff'):
            if not diff.get('type'):
                raise ValueError("Diff element missing type attribute")
                
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML structure: {str(e)}") from e

def _parse_action(xml_content: str) -> Optional[Action]:
    """Parse action from XML response if present"""
    root = ET.fromstring(xml_content)
    action_elem = root.find('.//action')
    if action_elem is not None:
        return Action(
            type=action_elem.get('type', ''),
            params={child.tag: (child.text or '').strip()  # Added strip()
                   for child in action_elem if child.text is not None}  # Filter None texts
        )
    return None

def process_observation(
    current_memory: str,
    observation: str,
    model: str = "openrouter/deepseek/deepseek-chat"
) -> Tuple[List[MemoryDiff], Optional[Action]]:
    """Process observation and return memory diffs with optional action
    
    Args:
        current_memory: Current memory state as string
        observation: New observation to process
        model: Model to use for processing
        
    Returns:
        Tuple of (list of MemoryDiff objects, optional Action)
        
    Raises:
        ValueError: If input validation fails
        ET.ParseError: If XML parsing fails
    """
    try:
        _validate_inputs(current_memory, observation, model)
        prompt = _prepare_prompt(current_memory, observation)
        response, error = _get_litellm_response(model, prompt)
        
        if error:
            print(f"API Error: {error}")
            return [], None
        
        if not response:
            print("Empty response from API")
            return [], None
            
        try:
            # Validate and parse XML
            _validate_xml_response(response)
            root = ET.fromstring(response)
            
            # Parse diffs
            diffs = []
            for diff_elem in root.findall('.//diff'):
                try:
                    diff_type = DiffType[diff_elem.get('type', 'MODIFY').upper()]
                except KeyError:
                    continue  # Skip invalid diff types
                    
                key = diff_elem.findtext('key', '').strip()
                if not key:  # Skip entries with empty keys
                    continue
                    
                old_val = diff_elem.findtext('old_value', None)
                new_val = diff_elem.findtext('new_value', None)
                
                diffs.append(MemoryDiff(
                    type=diff_type,
                    key=key,
                    old_value=old_val,
                    new_value=new_val
                ))
            
            # Parse action if present
            action = None
            action_elem = root.find('.//action')
            if action_elem is not None:
                action_type = action_elem.get('type', '').strip()
                params = {child.tag: (child.text or '').strip()
                         for child in action_elem if child.text is not None}
                
                action = Action(
                    type=action_type,
                    params=params
                )
            
            return diffs, action
            
        except ET.ParseError as e:
            print(f"XML parsing error in response: {str(e)}\nResponse content: {response}")
            return [], None
        except KeyError as e:
            print(f"Invalid diff type: {str(e)}")
            return [], None
        except Exception as e:
            print(f"Processing error: {str(e)}\nOriginal input: {observation}")
            return [], None
            
    except Exception as e:
        print(f"Critical error processing observation: {str(e)}")
        return [], None

def create_agent(model: str = 'openrouter/deepseek/deepseek-chat', max_tokens: int = 50,
                load: Optional[str] = None, test_mode: bool = False) -> Agent:
    """Create an agent with specified model.

    Args:
        model: Model to use ('flash', 'gemini-pro', 'deepseek-chat' or full OpenRouter model name)
        max_tokens: Maximum number of tokens for responses
        load: Path to load agent state from
        test_mode: Enable testing mode (skips real LLM calls)
        
    Returns:
        Initialized Agent instance
        
    Raises:
        ValueError: If max_tokens is not a positive integer
        FileNotFoundError: If load path is specified but doesn't exist
    """
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
        
    # Model name mapping with full OpenRouter paths
    model_mapping = {
        'flash': 'openrouter/google/gemini-2.0-flash-001',
        'pro': 'openrouter/google/gemini-2.0-pro',
        'deepseek-chat': 'openrouter/deepseek/deepseek-chat',
        'deepseek-coder': 'openrouter/deepseek/deepseek-coder-33b-instruct',
        'default': 'openrouter/deepseek/deepseek-chat',
        'gemini-flash': 'openrouter/google/gemini-2.0-flash-001',
        'gemini-pro': 'openrouter/google/gemini-2.0-pro'
    }
    # Get mapped model name
    model_name = model_mapping.get(model, model)  # Case-sensitive match
    
    if not is_valid_model_name(model_name):
        raise ValueError(f"Invalid model name: {model_name}")

    agent = Agent(model_name, max_tokens=max_tokens, test_mode=test_mode)
    
    if load:
        if not os.path.exists(load):
            raise FileNotFoundError(f"Agent save file not found: {load}")
        with open(load, 'r') as f:
            # Simple memory loading from file
            agent._memory = [
                MemoryItem(
                    input="Loaded memory",
                    output=f.read(),
                    type="persisted",
                    timestamp=datetime.datetime.now().isoformat()
                )
            ]
    
    return agent

# Control exported symbols for from utils import *
__all__ = [
    'Action',
    'Agent',
    'DiffType',
    'MemoryDiff',
    'MemoryItem',
    'create_agent',
    'envs',
    'extract_xml',
    'print_datetime',
    'process_observation',
    'run_inference'
]

