from memdiff.memory import process_observation
import litellm
import os

# Configure LiteLLM 
os.environ['LITELLM_LOG'] = 'DEBUG'  # Replace deprecated set_verbose

# Example usage
current_memory = "<current_state></current_state>"
observation = "User requested to update config values"

diffs, action = process_observation(current_memory, observation)

print("Memory Diffs:")
for diff in diffs:
    print(f"File: {diff.file_path}")
    print(f"Change: {diff.search} -> {diff.replace}")

if action:
    print(f"\nAction: {action.name}")
    for param, value in action.params.items():
        print(f"{param}: {value}")
