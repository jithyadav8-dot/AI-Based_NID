import json

with open("models/threshold_ae.json") as f:
    cfg = json.load(f)

cfg["threshold_previous"] = cfg["threshold"]
cfg["threshold"]          = 3.5
cfg["threshold_source"]   = "local_calibration_v3"
cfg["calibration_note"]   = (
    "Raised from 2.5 to 3.5 — Azure HTTPS flows produce AE error ~3.1 "
    "at benign p95. Threshold now sits above HTTPS noise while remaining "
    "sensitive to genuine attack traffic."
)

with open("models/threshold_ae.json", "w") as f:
    json.dump(cfg, f, indent=2)

print("Threshold set to 3.5")
