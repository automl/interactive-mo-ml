from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json

cs = ConfigurationSpace({"a": [1, 2, 3]})

cs_string = cs_json.write(cs)
with open('configspace.json', 'w') as f:
     f.write(cs_string)

with open('configspace.json', 'r') as f:
    json_string = f.read()
    config = cs_json.read(json_string)

print(config)