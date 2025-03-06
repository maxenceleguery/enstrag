from datasets import Dataset
import os
import json

json_dir = os.path.dirname(os.path.abspath(__file__))

data = []

for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(json_dir, filename)
        with open(filepath, 'r') as f:
            data.extend(json.load(f))

for question in data:
    assert 'Question' in question.keys()
    assert 'Chunks' in question.keys()
    assert len(question["Chunks"]) > 0
    assert all("chunk" in question["Chunks"][i] for i in range(len(question["Chunks"])))
    for chunk in question["Chunks"]:
        assert isinstance(chunk["chunk"], str)
        if isinstance(chunk["metadata"], list):
            chunk["metadata"] = chunk["metadata"][0]
            chunk["metadata"]["page"] = int(chunk["metadata"]["page"])

filepath = os.path.join(json_dir, "final_dataset.json")
with open(filepath, 'w') as f:
    json.dump(data, f)

dataset = Dataset.from_list(data)

dataset.push_to_hub('Maxenceleguery/enstrag_dataset')