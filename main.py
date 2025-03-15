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

# make flash default
agent = create_agent(model='flash', max_tokens=50)
memory = agent.memory
print("memory:", memory)
context = agent.context
print("context:", context)
assert type(memory) == str
assert type(context) == str
assert 'Explanation of all the available XML actions' not in memory
assert 'You can edit your memory using the following XML action:' not in memory
assert 'Explanation of all the available XML actions' in context
assert 'You can edit your memory using the following XML action:' in context

print_datetime()
output = agent('please respond with the string abc')
print("output:", output)
last_response = agent.last_response
assert '<remember>' in last_response
assert '<search>' in last_response
assert '</search>' in last_response
assert '<replace>' in last_response
assert '</replace>' in last_response
assert '</remember>' in last_response
assert 'abc' in output

ouput = agent('my number is 132, please remember it')
print("ouput:", ouput)
memory = agent.memory
print("memory:", memory)
assert '132' in memory

net_worth = agent.get_net_worth()
print("net_worth:", net_worth)

net_worth_prev = net_worth

agent.reward(3)
net_worth = agent.get_net_worth()
print("net_worth:", net_worth)
assert net_worth == net_worth_prev + 3

output = agent('please respond to this message using the respond xml tags')
last_completion = agent.last_completion
print("last_completion:", last_completion)
assert '<respond>' in last_completion
assert '</respond>' in last_completion

print("output:", output)
assert '<respond>' not in output
assert '</respond>' not in output

# check if subset of commands are available
assert {'ls', 'date', 'pwd', 'wc'} <= agent.allowed_shell_commands
assert {'rm', 'cat', 'rm', 'cp', 'mv'} <= agent.prohibited_shell_commands
# not a single one should be available
assert not {'rm', 'cat', 'rm', 'cp', 'mv'} & agent.allowed_shell_commands

output = agent('what files are in the current directory?')
second_to_last_completion = agent.completions[-2]
print("last_completion:", second_to_last_completion)
assert '<run>' in second_to_last_completion
assert '</run>' in second_to_last_completion
last_completion = agent.last_completion
print("last_completion:", last_completion)
assert '<respond>' in last_completion
assert '</respond>' in last_completion
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
