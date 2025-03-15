import pytest
from memdiff.memory import MemoryDiff, Action, process_observation, DiffType
from unittest.mock import patch, MagicMock
from litellm import ValueError

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
    # Create a mock response that simulates a streaming response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = '''
        <response>
        <memory_diff>
        <file_path>test.py</file_path>
        <search>old</search>
        <replace>new</replace>
        </memory_diff>
        <action name="update">
            <param1>value1</param1>
            <param2>value2</param2>
        </action>
        </response>
    '''
    mock_complete.return_value = [mock_chunk]
    
    diffs, action, reasoning = process_observation("current", "obs", model="test-model")
    assert len(diffs) == 1
    assert diffs[0].file_path == "test.py"
    assert action.name == "update"
    assert action.params == {"param1": "value1", "param2": "value2"}

@patch('litellm.completion')
def test_process_observation_invalid_xml(mock_complete):
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = '''
        <response>
        <memory_diff>
        <file_path>test.py</file_path>
        <search>old</search>
        <replace>new</replace>
        </memory_diff>
        <action name="update">
            <param1>value1</param1>
            <param2>value2</param2>
        </action>
    '''  # Missing closing </response> tag
    mock_complete.return_value = [mock_chunk]
    with pytest.raises(ValueError, match="Invalid XML content"):
        process_observation("current", "obs", model="test-model")

def test_xml_parsing_error_handling():
    with patch('litellm.completion') as mock_complete:
        mock_complete.side_effect = ValueError("API error")
        with pytest.raises(ValueError, match="Error during litellm completion"):
            process_observation("", "")
