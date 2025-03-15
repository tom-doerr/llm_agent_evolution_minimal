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
    
    # Extract memory diffs
    diff_matches = re.findall(r'<memory_diff>(.*?)</memory_diff>', xml_content, re.DOTALL)
    if diff_matches:
        # Parse diffs here...
        pass
        
    # Extract action
    action_match = re.search(r'<action name="([^"]+)"(.*?)>(.*?)</action>', xml_content, re.DOTALL)
    if action_match:
        # Parse action params here...
        pass

    return memory_diffs, action
