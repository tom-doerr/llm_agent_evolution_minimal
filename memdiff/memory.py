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
    type: DiffType = DiffType.SEARCH_REPLACE

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

def _prepare_prompt(current_memory: str, observation: str) -> str:
    return f"""Current memory state:
{current_memory}

New observation:
{observation}

Respond STRICTLY in this XML format:
<response>
  <memory_diff>
    <!-- 1+ SEARCH/REPLACE diffs -->
    <file_path>filename.ext</file_path>
    <search>EXACT existing code/text</search>
    <replace>NEW code/text</replace>
  </memory_diff>
  <action name="action_name">
    <!-- 0+ parameters -->
    <param_name>value</param_name>
  </action>
</response>

Examples:
1. File edit with action:
<response>
  <memory_diff>
    <file_path>config.py</file_path>
    <search>DEFAULT_MODEL = 'deepseek/deepseek-reasoner'</search>
    <replace>DEFAULT_MODEL = 'openrouter/google/gemini-2.0-flash-001'</replace>
  </memory_diff>
  <action name="reload_config">
    <module>config</module>
  </action>
</response>

2. Multiple files:
<response>
  <memory_diff>
    <file_path>app.py</file_path>
    <search>debug=True</search>
    <replace>debug=False</replace>
    <file_path>README.md</file_path>
    <search>Old feature list</search>
    <replace>New feature list</replace>
  </memory_diff>
</response>"""

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
            try:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        xml_content += delta.content
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
    except Exception as e:
        raise ValueError(f"Error during litellm completion: {e}") from e
    return xml_content, reasoning_content

def _parse_memory_diffs(xml_content: str) -> List[MemoryDiff]:
    memory_diffs = []
    diff_sections = re.findall(r'<memory_diff>(.*?)</memory_diff>', xml_content, re.DOTALL)
    for section in diff_sections:
        file_diffs = re.finditer(
            r'<file_path>(.*?)</file_path>\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>',
            section,
            re.DOTALL
        )
        for match in file_diffs:
            file_path = match.group(1).strip()
            search = match.group(2).strip()
            replace = match.group(3).strip()
            memory_diffs.append(
                MemoryDiff(file_path=file_path, search=search, replace=replace)
            )
    return memory_diffs

def _parse_action(xml_content: str) -> Optional[Action]:
    action_match = re.search(
        r'<action name="([^"]+)">(.*?)</action>',
        xml_content,
        re.DOTALL
    )
    if action_match:
        action_name = action_match.group(1)
        params = {}
        param_matches = re.findall(
            r'<([^>]+)>(.*?)</\1>',
            action_match.group(2),
            re.DOTALL
        )
        for param_name, param_value in param_matches:
            params[param_name.strip()] = param_value.strip()
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
