import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "annotations" / "mini_pixmo.jsonl"
OUTPUT = ROOT / "data" / "annotations"

with open(SOURCE, encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f if line.strip()
    ]

random.Random(42).shuffle(records)

split_index = int(len(records) * 0.8)

train = records[:split_index]
validation = records[split_index:]

for filename, subset in [
    ("train.jsonl", train),
    ("val.jsonl", validation)
]:

    with open(OUTPUT / filename, "w", encoding="utf-8") as f:

        for record in subset:
            f.write(json.dumps(record) + "\n")

print("Dataset split completed!")
print("Training images:", len(train))
print("Validation images:", len(validation))