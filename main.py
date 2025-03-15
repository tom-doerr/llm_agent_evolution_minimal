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
assert 'You can edit your memory using the following xml action:' in memory

print_datetime()
output = agent('please respond with the string abc')
print("output:", output)
assert 'abc' in output

ouput = agent('my number is 132, please remember it')
print("ouput:", ouput)
memory = agent.memory

net_worth = agent.get_net_worth()
print("net_worth:", net_worth)

net_worth_prev = net_worth

agent.reward(3)
net_worth = agent.get_net_worth()
print("net_worth:", net_worth)
assert net_worth == net_worth_prev + 3


