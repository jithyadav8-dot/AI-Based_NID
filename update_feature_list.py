# update_feature_list.py

import json

with open("processed/feature_list.json") as f:
    features = json.load(f)

print(f"Before: {len(features)} features")
print(f"Src Port in list: {'Src Port' in features}")

if "Src Port" in features:
    features.remove("Src Port")

with open("processed/feature_list.json", "w") as f:
    json.dump(features, f, indent=2)

print(f"After: {len(features)} features")
print("Saved: processed/feature_list.json")
