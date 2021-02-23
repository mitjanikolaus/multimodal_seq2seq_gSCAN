import json
import random

data = json.load(open("data/teacher/dataset.txt"))
print(len(data["examples"]["train"]))

data["examples"]["train"] = random.sample(data["examples"]["train"], 386000)

print(len(data["examples"]["train"]))

with open("data/teacher_small_50/dataset.txt", 'w') as outfile:
    json.dump(data, outfile)

print("success")


