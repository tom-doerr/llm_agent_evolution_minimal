import pytest
from memory import MemoryDiff, Action, process_observation, DiffType
from unittest.mock import patch

def test_memory_diff_initialization():
    diff = MemoryDiff(
        file_path="test.py",
        search="old",
        replace="new",
        type=DiffType.SEARCH_REPLACE
    )
    assert diff.file_path == "test.py"
    assert diff.search == "old"
    assert diff.replace == "new"

@patch('litellm.completion')
def test_process_observation_valid_response(mock_complete):
    mock_complete.return_value = [{
        'choices': [{'delta': {'content': '''
            <response>
            <memory_diff>
            <file>test.py</file>
            <search>old</search>
            <replace>new</replace>
            </memory_diff>
            </response>
        '''}}]
    }]
    
    diffs, action = process_observation("current", "obs")
    assert len(diffs) == 1
    assert diffs[0].file_path == "test.py"

def test_xml_parsing_error_handling():
    with patch('litellm.completion') as mock_complete:
        mock_complete.side_effect = Exception("API error")
        diffs, action = process_observation("", "")
        assert diffs == []
        assert action is None
