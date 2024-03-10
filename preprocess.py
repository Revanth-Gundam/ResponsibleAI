import json

with open('benchmark/data/zero_context/nq.json', 'r') as f:
    data = json.load(f)

with open('/data/circulars/DATA/CircularsGPT/pure_language/RefChecker/benchmark/human_annotations_v1/zero_context/nq_alpaca_7B_answers.json', 'r') as f:
    data2 = json.load(f)

print(data[0].keys())
print("====================================")
print(data2[0].keys())

# data_f = []

# for element in data2:
#     new = {}
#     new = {'id': element['id'], 'input': element['input'], 'response': element['response'], 'triplets': element['triplets']}

#     id_1 = element['id']

#     # Look for the same id in data
#     for element_2 in data:
#         id_2 = element_2['id']
#         if id_1 == id_2:
#             new['context'] = element_2['context']
#             break

#     data_f.append(new)

# with open('2.json', 'w') as f:
#     json.dump(data_f, f, indent=4)

data_f = []

for element in data2:
    entry = {}

    id_1 = element['id']
    entry['id'] = id_1
    entry['response'] = element['response']
    entry['triplets'] = [t["triplet"] for t in element['claude2_response_kg']]

    # Find the triplets
    for element_2 in data:
        id_2 = element_2['id']
        if id_1 == id_2:
            entry["context"] = element_2["context"]
            break

    print(entry)
    
    data_f.append(entry)

with open('alpaca_7b_checker.json', 'w') as f:
    json.dump(data_f, f, indent=4)
