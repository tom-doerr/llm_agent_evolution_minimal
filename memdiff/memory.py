from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional, Dict, List, Tuple
import litellm

import xml.etree.ElementTree as ET
from litellm import completion

class DiffType(Enum):
    SEARCH_REPLACE = "search/replace"
    ACTION = "action"

@dataclass
class MemoryDiff:
    file_path: str
    search: str
    replace: str
    type: DiffType

@dataclass
class Action:
    name: str
    params: Dict[str, str]

def _validate_inputs(current_memory: str, observation: str, model: str) -> None:
    """Validates the inputs to the process_observation function."""
    if not isinstance(current_memory, str):
        raise ValueError("current_memory must be a string")
    if not isinstance(observation, str):
        raise ValueError("observation must be a string")
    if not isinstance(model, str):
        raise ValueError("model must be a string")

def _create_prompt_header(current_memory: str, observation: str) -> str:
    """Creates the header for the prompt."""
    return f"""Current memory state:\n{current_memory}\n\nNew observation:\n{observation}\n\nRespond STRICTLY in this XML format:\n"""

def _create_prompt_body() -> str:
    """Creates the body of the prompt."""
    return """<response>\n  <memory_diff>\n    <!-- 1+ SEARCH/REPLACE diffs -->\n    <file_path>filename.ext</file_path>\n    <search>EXACT existing code/text</search>\n    <replace>NEW code/text</replace>\n  </memory_diff>\n  <action name="action_name">\n    <!-- 0+ parameters -->\n    <param_name>value</param_name>\n  </action>\n</response>\n"""

def _create_prompt_examples() -> str:
    """Creates example responses for the prompt."""
    example1 = """1. File edit:\n<response>\n  <memory_diff>\n    <file_path>config.py</file_path>\n    <search>DEFAULT_MODEL = 'deepseek/deepseek-reasoner'</search>\n    <replace>DEFAULT_MODEL = 'openrouter/gemini'</replace>\n  </memory_diff>\n  <action name="reload_config">\n    <module>config</module>\n  </action>\n</response>\n"""
    example2 = """2. Multiple files:\n<response>\n  <memory_diff>\n    <file_path>app.py</file_path>\n    <search>debug=True</search>\n    <replace>debug=False</replace>\n    <file_path>README.md</file_path>\n    <search>Old feature list</search>\n    <replace>New feature list</replace>\n  </memory_diff>\n</response>"""
    return "Examples:\n" + example1 + example2

def _prepare_prompt(current_memory: str, observation: str) -> str:
    """Combines the header, body, and examples to create the full prompt."""
    header = _create_prompt_header(current_memory, observation)
    body = _create_prompt_body()
    examples = _create_prompt_examples()
    return header + body + examples

def _process_chunk(stream_chunk) -> Tuple[str, str]:
    """Processes a chunk of the stream response from the LLM."""
    xml_content = ""
    reasoning_content = ""
    try:
        if stream_chunk.choices and stream_chunk.choices[0].delta:
            delta = stream_chunk.choices[0].delta
            xml_content = delta.content if hasattr(delta, 'content') and delta.content else ""
            reasoning_content = delta.reasoning_content if hasattr(delta, 'reasoning_content') and delta.reasoning_content else ""
    except Exception as e:
        print(f"Error processing chunk: {e}")
    return xml_content, reasoning_content

def _get_litellm_response(model: str, prompt: str) -> Tuple[str, str]:
    """Gets the response from the litellm completion API."""
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        xml_content, reasoning_content = "",""
        for chunk in response:
            xml_update, reasoning_update = _process_chunk(chunk)
            xml_content += xml_update
            reasoning_content += reasoning_update
        return xml_content.strip(), reasoning_content.strip()
    except Exception as e:
        raise ValueError(f"Error during litellm completion: {e}") from e

def _extract_file_diffs(section: str) -> List[MemoryDiff]:
    """Extracts file diffs from a memory_diff section."""
    file_diffs = []
    matches = re.finditer(
        r'<file_path>(.*?)</file_path>\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>',
        section,
        re.DOTALL
    )
    for match in matches:
        file_path = match.group(1).strip()
        search = match.group(2).strip()
        replace = match.group(3).strip()
        file_diffs.append(
            MemoryDiff(file_path=file_path, search=search, replace=replace, type=DiffType.SEARCH_REPLACE)
        )
    return file_diffs

def _parse_memory_diffs(xml_content: str) -> List[MemoryDiff]:
    """Parses the memory diffs from the XML content."""
    memory_diffs: List[MemoryDiff] = []
    diff_sections = re.findall(r'<memory_diff>(.*?)</memory_diff>', xml_content, re.DOTALL)
    for memory_diff_section in diff_sections:
        memory_diffs.extend(_extract_file_diffs(memory_diff_section))
    return memory_diffs

def _parse_action_params(action_content: str) -> Dict[str, str]:
    """Parses the action parameters from the action content."""
    params = {}
    param_matches = re.findall(r'<([^>]+)>(.*?)</\1>', action_content, re.DOTALL)
    for param_name, param_value in param_matches:
        params[param_name.strip()] = param_value.strip()
    return params

def _parse_action(xml_content: str) -> Optional[Action]:
    """Parses the action from the XML content."""
    action_match = re.search(r'<action name="([^"]+)">(.*?)</action>', xml_content, re.DOTALL)
    if action_match:
        action_name = action_match.group(1)
        params = _parse_action_params(action_match.group(2))
        return Action(name=action_name, params=params)
    return None

def _validate_xml_response(xml_content: str) -> None:
    """Validates that the XML response is well-formed and contains the root <response> tag."""
    if not xml_content.strip().startswith("<response>"):
        raise ValueError("Invalid XML response format - missing root <response> tag")
    try:
        ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML content: {e}") from e

def process_observation(
    current_memory: str, 
    observation: str,
    model: str = "deepseek/deepseek-reasoner"
) -> Tuple[List[MemoryDiff], Optional[Action], str]:
    """Processes an observation and returns memory diffs, an action, and reasoning content."""
    _validate_inputs(current_memory, observation, model)

    prompt = _prepare_prompt(current_memory, observation)
    xml_content, reasoning_content = _get_litellm_response(model, prompt)

    _validate_xml_response(xml_content)
    memory_diffs = _parse_memory_diffs(xml_content)
    action = _parse_action(xml_content)
    return memory_diffs, action, reasoning_content
