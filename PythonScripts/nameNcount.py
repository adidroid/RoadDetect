import json
from collections import Counter
path = "coco/train.json"

with open(path) as f:
    data = json.load(f)

# List all category names and IDs
categories = {cat["id"]: cat["name"] for cat in data["categories"]}

print("Number of classes:", len(categories))
print("Classes:", categories)

cls_count = Counter()
for ann in data["annotations"]:
    cls_count[ann["category_id"]] += 1

print("Class Distribution:")
for cls_id, count in cls_count.items():
    print(categories[cls_id], ":", count)