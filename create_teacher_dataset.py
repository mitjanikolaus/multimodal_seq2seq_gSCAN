import json

data = json.load(open("train_teacher/dataset.txt"))
print(len(data["examples"]["test"]))

data_comp = json.load(open("data/compositional_splits/dataset.txt"))

data_comp["examples"]["train"] += data["examples"]["test"]
print(len(data_comp["examples"]["train"]))

with open("data/teacher/dataset.txt", 'w') as outfile:
    json.dump(data_comp, outfile)

print("success")


