import time
start = time.time()
from utils import run_inference, create_agent, print_datetime
end = time.time()
duration = end - start
print("duration:", duration)
assert duration < 0.5

print('test', flush=True)

# start = time.time()
# # use the flash model by default
# response = run_inference("Hi!", model='flash')
# end = time.time()
# assert end - start < 5, 'reduce max tokens'
# print("response:", response)
response = ''

from utils import extract_xml
extracted_xml_data = extract_xml(response)
print("extracted_xml_data:", extracted_xml_data)

# MODEL = 'flash'
# MODEL = 'openrouter/deepseek/deepseek-chat'
MODEL = 'deepseek/deepseek-chat'
agent = create_agent(model=MODEL, max_tokens=50)
memory = agent.memory
print("memory:", memory)
context = agent.context
print("context:", context)
assert type(memory) == str
assert type(context) == str
assert 'Explanation of all the available XML actions' in context
assert 'You can edit your memory using the following XML action:' in context
assert 'Explanation of all the available XML actions' not in memory
assert 'You can edit your memory using the following XML action:' not in memory

assert 'Examples of how to use the XML actions:' in context
assert 'You can use multiple actions in a single completion' in context
length_context = len(context)
print("length_context:", length_context)
assert length_context > 1000

print_datetime()
output = agent('please respond with the string abc')
print("output:", output)
assert 'abc' in output

# don't just add all messages to the memory
# only the agent should be able to add to the memory
# don't mock llm calls, just run it. not that expensive
ouput = agent('my number is 132, please remember it')
print("ouput:", ouput)
last_completion = agent.last_completion
print("last_completion:", last_completion)
memory = agent.memory
print("memory:", memory)
assert '<remember>' in last_completion
assert '<search>' in last_completion
assert '</search>' in last_completion
assert '<replace>' in last_completion
assert '</replace>' in last_completion
assert '</remember>' in last_completion
assert '132' in memory

net_worth = agent.get_net_worth()
print("net_worth:", net_worth)

net_worth_prev = net_worth

agent.reward(3)
net_worth = agent.get_net_worth()
print("net_worth:", net_worth)
assert net_worth == net_worth_prev + 3

output = agent('please respond to this message using the message xml tags')
last_completion = agent.last_completion
print("last_completion:", last_completion)
assert '<message>' in last_completion
assert '</message>' in last_completion

print("output:", output)
assert '<message>' not in output
assert '</message>' not in output

# check if subset of commands are available
assert {'ls', 'date', 'pwd', 'wc'} <= agent.allowed_shell_commands
assert {'rm', 'cat', 'rm', 'cp', 'mv'} <= agent.prohibited_shell_commands
# not a single one should be available
assert not {'rm', 'cat', 'rm', 'cp', 'mv'} & agent.allowed_shell_commands

output = agent('what files are in the current directory?')
second_to_last_completion = agent.completions[-2]
print("last_completion:", second_to_last_completion)
assert '<shell>' in second_to_last_completion
assert '</shell>' in second_to_last_completion
last_completion = agent.last_completion
print("last_completion:", last_completion)
assert '<message>' in last_completion
assert '</message>' in last_completion
print("output:", output)
assert 'plexsearch.log' in output

# file editing testing
with open('test.txt', 'w') as f:
    f.write('this is a test\ntest\ntest\nfile with the text abcd in it')
output = agent('please remove the text "abcd" from the file "test.txt"')
print("output:", output)
last_completion = agent.last_completion
print("last_completion:", last_completion)
assert '<edit>' in last_completion
assert '<search>' in last_completion
assert '</search>' in last_completion
assert '<replace>' in last_completion
assert '</replace>' in last_completion
assert '</edit>' in last_completion
with open('test.txt', 'r') as f:
    content = f.read()

assert 'abcd' not in content


agent_a = create_agent(model=MODEL, max_tokens=50)
agent_b = create_agent(model=MODEL, max_tokens=50)
output_a = agent_a('please remember my secret number 321!')
print("output_a:", output_a)
last_completion_a = agent_a.last_completion
print("last_completion_a:", last_completion_a)
output_b = agent_b('please remember my secret number 477!')
print("output_b:", output_b)
last_completion_b = agent_b.last_completion
print("last_completion_b:", last_completion_b)

memory_a = agent_a.memory
print("memory_a:", memory_a)
memory_b = agent_b.memory
print("memory_b:", memory_b)

assert '321' in memory_a
assert '477' in memory_b

agent_a.reward(1,000,000)

net_worth_a = agent_a.get_net_worth()
print("net_worth_a:", net_worth_a)
net_worth_b = agent_b.get_net_worth()
print("net_worth_b:", net_worth_b)

net_worth_a_prev = net_worth_a
net_worth_b_prev = net_worth_b

from utils import base_env_manager

mating_cost = base_env_manager.mating_cost
print("mating_cost:", mating_cost)

agent_c = agent_a.mate(agent_b) 
memory_c = agent_c.memory
print("memory_c:", memory_c)
assert '321' in memory_c
assert '477' in memory_c


net_worth_a = agent_a.get_net_worth()
print("net_worth_a:", net_worth_a)
net_worth_b = agent_b.get_net_worth()
print("net_worth_b:", net_worth_b)

assert net_worth_a == net_worth_a_prev - mating_cost
assert net_worth_b == net_worth_b_prev

from utils import envs

a_env = envs['a_env']
reward = a_env('')
assert reward == 0

reward = a_env('aaaaa')
assert reward == 5

reward = a_env('abaaaccccc')
assert reward == 4

reward = a_env('abaaacccccaa')
assert reward == 6

total_num_completions = agent_a.total_num_completions
print("total_num_completions:", total_num_completions)
total_num_completions_prev = total_num_completions

net_worth_a = agent_a.get_net_worth()
print("net_worth_a:", net_worth_a)
net_worth_a_prev = net_worth_a

reward = a_env(agent_a)
print("reward:", reward)

total_num_completions = agent_a.total_num_completions
print("total_num_completions:", total_num_completions)
assert total_num_completions == total_num_completions_prev + 1

net_worth_a = agent_a.get_net_worth()
print("net_worth_a:", net_worth_a)
assert net_worth_a == net_worth_a_prev + reward
net_worth_a_prev = net_worth_a

memory_a = agent_a.memory
print("memory_a:", memory_a)
memory_a_prev = memory_a
agent_a.save('agent_a.toml')

agent_a = create_agent(model='flash', max_tokens=50, load='agent_a.toml')
memory_a = agent_a.memory
print("memory_a:", memory_a)
assert memory_a == memory_a_prev

assert net_worth_a == net_worth_a_prev

