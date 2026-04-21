import csv
import random

random.seed(7)

# Read the gold label sheet
rows = []
with open('data/form300_factory/manifests/gold_label_sheet.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Sample 5 rows from different field families (just for display)
sampled = []
for family in ['short_id', 'date', 'name_text']:
    for row in rows:
        if row['field_family'] == family:
            sampled.append(row)
            break

print("Sample rows to label:")
for row in sampled:
    print(f"crop_id: {row['crop_id']}")
    print(f"image_path: {row['image_path']}")
    print(f"field_name: {row['field_name']}")
    print(f"field_family: {row['field_family']}")
    print("----")