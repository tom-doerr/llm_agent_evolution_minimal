from memdiff.memory import process_observation
import litellm
import os

# Configure LiteLLM 
os.environ['LITELLM_LOG'] = 'DEBUG'  # Replace deprecated set_verbose

# Example usage with CLI arguments
import argparse

parser = argparse.ArgumentParser(description='Process memory diffs')
parser.add_argument('--model', default='deepseek/deepseek-reasoner',
                    help='Model to use (deepseek/deepseek-reasoner or openrouter/google/gemini-2.0-flash-001)')
parser.add_argument('--observation', default='User requested to update config values',
                    help='Observation text to process')
args = parser.parse_args()

current_memory = "<current_state></current_state>"

diffs, action = process_observation(current_memory, observation)

print("Memory Diffs:")
for diff in diffs:
    print(f"File: {diff.file_path}")
    print(f"Change: {diff.search} -> {diff.replace}")

if action:
    print(f"\nAction: {action.name}")
    for param, value in action.params.items():
        print(f"{param}: {value}")
