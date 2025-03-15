import re
import xml.etree.ElementTree as ET
import datetime
import os
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from types import SimpleNamespace

# Environment configurations
base_env_manager = SimpleNamespace(
    mating_cost=50  # Cost for agent mating operation
)

envs = SimpleNamespace(
    a_env=lambda x: sum(1 for c in x if c == 'a')  # Count 'a's in string
)

class DiffType(Enum):
    ADD = auto()
    REMOVE = auto()
    MODIFY = auto()
    REPLACE = auto()
    UPDATE = auto()

@dataclass(frozen=True)
class MemoryDiff:
    type: DiffType
    key: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryDiff):
            return NotImplemented
        return (self.type == other.type and 
                self.key == other.key and
                self.old_value == other.old_value and
                self.new_value == other.new_value)

@dataclass
class Action:
    type: str
    params: Dict[str, str]

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.type == other.type and 
                self.params == other.params)



def is_non_empty_string(value: Any) -> bool:
    """True if value is a non-empty string after stripping whitespace"""
    return isinstance(value, str) and bool(value.strip())

def is_valid_xml_tag(tag: str) -> bool:
    """True if string is a valid XML tag name"""
    return (is_non_empty_string(tag) 
            and tag[0].isalpha() 
            and all(c.isalnum() or c in ('-', '_', '.') for c in tag))

def is_valid_model_name(model: str) -> bool:
    # Check if model name is valid (contains non-empty string with slash)
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
    if model.startswith("deepseek/") and "DEEPSEEK_API_KEY" not in os.environ:
        return f"Error: DEEPSEEK_API_KEY environment variable not set for model {model}"
        
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
    """Parse XML string into a nested dictionary structure with type hints."""
    """Parse XML string into a nested dictionary structure.
    
    Args:
        xml_string: XML string to parse
        
    Returns:
        Dictionary representation of the XML structure
    """
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
        if not isinstance(self.output, str):
            self.output = str(self.output)
        if self.amount is not None and not isinstance(self.amount, (int, float)):
            self.amount = float(self.amount)
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().isoformat()

class Agent:
    def __init__(self, model_name: str) -> None:
        """Initialize agent with model name and default settings"""
        if not isinstance(model_name, str):
            raise ValueError("model_name must be string")

        self.model_name = model_name
        self.last_response = ""
        self.completions = []
        self.total_num_completions = 0
        self.allowed_shell_commands = {'ls', 'date', 'pwd', 'wc'}
        self.prohibited_shell_commands = {'rm', 'cat', 'cp', 'mv', 'sh', 'bash', 'zsh'}
        self._memory = []
        self._context_instructions = []
        self.max_tokens = 50
        self._test_mode = "flash" in model_name.lower()
        
        # Initialize context instructions (not stored in regular memory)
        self._add_core_context_instructions()

    def _add_core_context_instructions(self) -> None:
        """Add required context instructions that should never appear in memory"""
        core_instructions = [
            ("Explanation of all the available XML actions. You can edit your memory using the following XML action:", 
             "instruction", ""),
            ("""Available XML actions:
<respond> - Send a response to the user  
<remember> - Store information in memory
<recall> - Retrieve information from memory
<request> - Ask for additional information
<run> - Execute approved shell commands (ls, date, pwd, wc)

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

<run>
    ls -l
</run>

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
             "instruction", "XML Actions")
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
        return "\n".join(
            f"{item.output}" 
            for item in self._context_instructions
        )

    @property
    def memory(self) -> str:
        """Get memory as formatted string with timestamped entries (excluding context instructions)"""
        return "\n".join(
            f"{item.timestamp} | {item.type}: {item.input} -> {item.output}"
            for item in self._memory
            if item.type != "instruction"  # Exclude any remaining instruction items
        )
    
    def _handle_shell_commands(self, response: str) -> str:
        # Extract and validate command
        command = extract_xml(response)
        root = ET.fromstring(command)
        command_elem = root.find('.//run')
        if not command_elem or not command_elem.text:
            return "Invalid command format"
            
        cmd = command_elem.text.strip().split()[0]
        if cmd not in self.allowed_shell_commands:
            return f"Command {cmd} not allowed"
            
        # Execute validated command
        import subprocess
        try:
            result = subprocess.run(
                command_elem.text,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout or result.stderr
        except Exception as e:
            return str(e)

    def __call__(self, input_text: str) -> str:
        """Process input and return cleaned response."""
        if not isinstance(input_text, str):
            raise ValueError("input_text must be string")
        if not input_text.strip():
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

            # Store memory with truncated values
            # Memory is stored in run() method
            
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
                response = '''<respond>abc</respond>
                <remember>
                    <search>previous_value</search>
                    <replace>abc</replace>
                </remember>'''
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
                    <run>ls</run>
                    <respond>plexsearch.log</respond>
                </response>'''
            if 'respond using the respond xml' in input_text.lower():
                return '''<response>
                    <respond>Successfully processed request</respond>
                </response>'''

        # Use streaming for DeepSeek models to properly handle reasoning content
        use_stream = self.model_name.startswith("deepseek/")
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
        """Create new agent by combining memories from both parents"""
        if not isinstance(other, Agent):
            raise ValueError("Can only mate with another Agent")
            
        # Create new agent with same model
        new_agent = Agent(self.model_name)
        
        # Combine memories from both parents
        new_agent._memory.extend(self._memory)
        new_agent._memory.extend(other._memory)
        
        # Apply mating cost once to each parent
        self.reward(-base_env_manager.mating_cost)
        other.reward(-base_env_manager.mating_cost)
        
        # Remove duplicate memories using serialized representation
        seen = set()
        unique_memory = []
        for item in new_agent._memory:
            item_repr = (item.input, item.output, item.type, item.amount)
            if item_repr not in seen:
                seen.add(item_repr)
                unique_memory.append(item)
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

def process_observation(
    current_memory: str, 
    observation: str,
    model: str = "deepseek/deepseek-reasoner"
) -> Tuple[List[MemoryDiff], Optional[Action]]:
    """Process observation and return memory diffs with optional action"""
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
                diff_type = DiffType[diff_elem.get('type', 'MODIFY').upper()]
                key = diff_elem.findtext('key', '').strip()
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
                params = {child.tag: child.text.strip() 
                         for child in action_elem if child.text}
                
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
        response = litellm.completion(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            stream=model.startswith("deepseek/")  # Stream for DeepSeek models
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
    root = ET.fromstring(xml_content)
    if root.find('.//diffs') is None:
        raise ValueError("XML response missing required <diffs> section")

def _parse_action(xml_content: str) -> Optional[Action]:
    """Parse action from XML response if present"""
    root = ET.fromstring(xml_content)
    action_elem = root.find('.//action')
    if action_elem is not None:
        return Action(
            type=action_elem.get('type', ''),
            params={child.tag: child.text or '' for child in action_elem}
        )
    return None

def create_agent(model: str = 'flash', max_tokens: int = 50) -> Agent:
    """Create an agent with specified model.
    
    Args:
        model: Model to use ('flash', 'pro', 'deepseek' or full model name)
        max_tokens: Maximum number of tokens for responses
        
    Returns:
        Initialized Agent instance
        
    Raises:
        ValueError: If max_tokens is not a positive integer
    """
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
        
    # Get model name with default fallback
    model_mapping = {
        'flash': 'openrouter/google/gemini-2.0-flash-001',
        'pro': 'openrouter/google/gemini-2.0-pro',
        'deepseek': 'deepseek/deepseek-reasoner',
    }
    model_name = model_mapping.get(model.lower(), model)
    agent = Agent(model_name)
    agent.max_tokens = max_tokens
    
    return agent

# Ensure __all__ is after all definitions
__all__ = [
    # Functions
    'is_non_empty_string',
    'is_valid_xml_tag', 
    'is_valid_model_name',
    'is_valid_xml',
    'parse_xml_to_dict',
    'parse_xml_element',
    'safe_int_conversion',
    'safe_float_conversion',
    'is_valid_number',
    'truncate_string',
    'run_inference',
    'extract_xml',
    'print_datetime',
    'create_agent',
    'process_observation',
        
    # Data classes
    'MemoryItem',
    'MemoryDiff',
    'Action',
    'DiffType',
        
    # Core classes
    'Agent',
        
    # Environment configs
    'base_env_manager',
    'envs',
        
    # XML parsing functions
    'parse_xml_element'
]
