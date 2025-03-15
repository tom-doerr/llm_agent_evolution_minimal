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

def process_observation(current_memory: str, observation: str) -> tuple[List[MemoryDiff], Optional[Action]]:
    # Prepare the prompt for the AI
    prompt = f"""Current memory:
{current_memory}

Observation:
{observation}

Output format:
<response>
<memory_diff>
<!-- SEARCH/REPLACE blocks -->
</memory_diff>
<action name="..."><param1>value</param1></action>
</response>"""

    # Get completion from LiteLLM
    response = litellm.completion(
        model="deepseek/deepseek-reasoner",
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
