from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional, Dict, List, Tuple
import litellm

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
    if not isinstance(current_memory, str):
        raise ValueError("current_memory must be a string")
    if not isinstance(observation, str):
        raise ValueError("observation must be a string")
    if not isinstance(model, str):
        raise ValueError("model must be a string")

def _create_prompt_header(current_memory: str, observation: str) -> str:
    return f"""Current memory state:\n{current_memory}\n\nNew observation:\n{observation}\n\nRespond STRICTLY in this XML format:\n"""

def _create_prompt_body() -> str:
    return """<response>\n  <memory_diff>\n    <!-- 1+ SEARCH/REPLACE diffs -->\n    <file_path>filename.ext</file_path>\n    <search>EXACT existing code/text</search>\n    <replace>NEW code/text</replace>\n  </memory_diff>\n  <action name="action_name">\n    <!-- 0+ parameters -->\n    <param_name>value</param_name>\n  </action>\n</response>\n"""

def _create_prompt_examples() -> str:
    return """Examples:\n1. File edit with action:\n<response>\n  <memory_diff>\n    <file_path>config.py</file_path>\n    <search>DEFAULT_MODEL = 'deepseek/deepseek-reasoner'</search>\n    <replace>DEFAULT_MODEL = 'openrouter/google/gemini-2.0-flash-001'</replace>\n  </memory_diff>\n  <action name="reload_config">\n    <module>config</module>\n  </action>\n</response>\n\n2. Multiple files:\n<response>\n  <memory_diff>\n    <file_path>app.py</file_path>\n    <search>debug=True</search>\n    <replace>debug=False</replace>\n    <file_path>README.md</file_path>\n    <search>Old feature list</search>\n    <replace>New feature list</replace>\n  </memory_diff>\n</response>"""

def _prepare_prompt(current_memory: str, observation: str) -> str:
    header = _create_prompt_header(current_memory, observation)
    body = _create_prompt_body()
    examples = _create_prompt_examples()
    return header + body + examples

def _process_chunk(chunk) -> Tuple[str, str]:
    xml_content = ""
    reasoning_content = ""
    try:
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                xml_content += delta.content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
    except Exception as e:
        print(f"Error processing chunk: {e}")
    return xml_content, reasoning_content

def _get_litellm_response(model: str, prompt: str) -> Tuple[str, str]:
    xml_content = ""
    reasoning_content = ""
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in response:
            xml_update, reasoning_update = _process_chunk(chunk)
            xml_content += xml_update
            reasoning_content += reasoning_update

    except Exception as e:
        raise ValueError(f"Error during litellm completion: {e}") from e
    return xml_content, reasoning_content

def _extract_file_diffs(section: str) -> List[MemoryDiff]:
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
    memory_diffs = []
    diff_sections = re.findall(r'<memory_diff>(.*?)</memory_diff>', xml_content, re.DOTALL)
    for section in diff_sections:
        memory_diffs.extend(_extract_file_diffs(section))
    return memory_diffs

def _parse_action_params(action_content: str) -> Dict[str, str]:
    params = {}
    param_matches = re.findall(r'<([^>]+)>(.*?)</\1>', action_content, re.DOTALL)
    for param_name, param_value in param_matches:
        params[param_name.strip()] = param_value.strip()
    return params

def _parse_action(xml_content: str) -> Optional[Action]:
    action_match = re.search(r'<action name="([^"]+)">(.*?)</action>', xml_content, re.DOTALL)
    if action_match:
        action_name = action_match.group(1)
        params = _parse_action_params(action_match.group(2))
        return Action(name=action_name, params=params)
    return None

def process_observation(
    current_memory: str, 
    observation: str,
    model: str = "deepseek/deepseek-reasoner"
) -> Tuple[List[MemoryDiff], Optional[Action], str]:
    _validate_inputs(current_memory, observation, model)
    prompt = _prepare_prompt(current_memory, observation)
    xml_content, reasoning_content = _get_litellm_response(model, prompt)

    if not xml_content.strip().startswith("<response>"):
        raise ValueError("Invalid XML response format - missing root <response> tag")

    memory_diffs = _parse_memory_diffs(xml_content)
    action = _parse_action(xml_content)

    return memory_diffs, action, reasoning_content
