import os
import json

# Папка, где лежат все attack_meta_*.json
input_folder = "logs/attack_meta"
output_file = "all_attacks.json"

all_attacks = []

for filename in os.listdir(input_folder):
    if filename.endswith(".json") and filename.startswith("attack_meta"):
        with open(os.path.join(input_folder, filename), "r") as f:
            data = json.load(f)
            all_attacks.append(data)

with open(output_file, "w") as f:
    json.dump(all_attacks, f, indent=4)

print(f"Объединено {len(all_attacks)} атак в файл {output_file}")
