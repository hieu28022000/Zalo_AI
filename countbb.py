import json

with open('output/SubmitFilev5.3.json', 'r') as f:
    data = json.load(f)

print(len(data))
