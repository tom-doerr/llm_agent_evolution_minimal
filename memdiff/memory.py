from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional, Dict, List
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

def process_observation(
    current_memory: str, 
    observation: str,
    model: str = "deepseek/deepseek-reasoner"
) -> tuple[List[MemoryDiff], Optional[Action]]:
    # Prepare the prompt for the AI with explicit formatting instructions
    prompt = f"""Current memory state:
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

    # Get completion from LiteLLM
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # Process streaming response
    xml_content = ""
    for chunk in response:
        xml_content += chunk.choices[0].delta.content or ""

    # Parse XML response
    memory_diffs = []
    action = None
    
    # Validate we got valid XML content
    if not xml_content.strip().startswith("<response>"):
        raise ValueError("Invalid XML response format - missing root <response> tag")
        
    # Parse memory diffs
    diff_sections = re.findall(r'<memory_diff>(.*?)</memory_diff>', xml_content, re.DOTALL)
    for section in diff_sections:
        # Find all file diffs in the section
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
            
    # Parse action
    action_match = re.search(
        r'<action name="([^"]+)">(.*?)</action>',
        xml_content,
        re.DOTALL
    )
    if action_match:
        action_name = action_match.group(1)
        params = {}
        # Parse params
        param_matches = re.findall(
            r'<([^>]+)>(.*?)</\1>',
            action_match.group(2),
            re.DOTALL
        )
        for param_name, param_value in param_matches:
            params[param_name.strip()] = param_value.strip()
        action = Action(name=action_name, params=params)

    return memory_diffs, action
