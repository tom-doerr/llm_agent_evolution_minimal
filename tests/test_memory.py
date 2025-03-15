import pytest
from memdiff.memory import MemoryDiff, Action, process_observation, DiffType
from typing import List, Optional, Tuple, Dict
from unittest.mock import patch, MagicMock
from litellm import ValueError

def test_memory_diff_initialization():
    """Tests the initialization of the MemoryDiff class."""
    diff = MemoryDiff(
        file_path="test.py",
        search="old",
        replace="new",
        type=DiffType.SEARCH_REPLACE
    )
    assert diff.file_path == "test.py"
    assert diff.search == "old"
    assert diff.replace == "new"

@patch('memdiff.memory.litellm.completion')
def test_process_observation_valid_response(mock_completion):
    """Tests process_observation with a valid XML response."""
    # Define the mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(delta=MagicMock(content='''
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
        ''', reasoning_content="Reasoning here"))]

    # Configure the mock to return the mock response when called
    mock_completion.return_value = [mock_response]

    # Call the function under test
    diffs, action, reasoning = process_observation("current", "obs", model="test-model")

    # Assert the results
    assert len(diffs) == 1
    assert diffs[0].file_path == "test.py"
    assert diffs[0].search == "old"
    assert diffs[0].replace == "new"
    assert action.name == "update"
    assert action.params == {"param1": "value1", "param2": "value2"}
    assert reasoning == "Reasoning here"

@patch('litellm.completion')
def test_process_observation_no_diffs_no_action(mock_complete):
    """Tests process_observation with no memory diffs and no action."""
    # Create a mock response with no memory diffs and no action
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = '<response></response>'
    mock_complete.return_value = [mock_chunk]

    diffs, action, reasoning = process_observation("current", "obs", model="test-model")
    assert len(diffs) == 0
    assert action is None

@patch('litellm.completion')
def test_process_observation_no_action(mock_complete):
    """Tests process_observation with memory diffs but no action."""
    # Create a mock response with memory diffs but no action
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
        </response>
    '''
    mock_complete.return_value = [mock_chunk]

    diffs, action, reasoning = process_observation("current", "obs", model="test-model")
    assert len(diffs) == 1
    assert diffs[0].file_path == "test.py"
    assert action is None


@patch('litellm.completion')
def test_process_observation_invalid_xml(mock_complete):
    """Tests process_observation with invalid XML, expecting a ValueError."""
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
    """Tests error handling when XML parsing fails."""
    with patch('litellm.completion') as mock_complete:
        mock_complete.side_effect = ValueError("API error")
        with pytest.raises(ValueError, match="Error during litellm completion"):
            process_observation("", "")
