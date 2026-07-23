# Unified 8-class label scheme used by both XGBoost and the autoencoder.
# Both datasets map into these classes before any modelling.

UNIFIED_LABELS = {
    "BENIGN":       0,
    "DoS":          1,
    "DDoS":         2,
    "PortScan":     3,
    "BruteForce":   4,
    "WebAttack":    5,
    "Botnet":       6,
    "Infiltration": 7,
}

# ── CSE-CIC-IDS2018 raw labels → unified ─────────────────────────
LABEL_MAP_2018 = {
    "Benign":                    "BENIGN",
    "DoS attacks-GoldenEye":     "DoS",
    "DoS attacks-Hulk":          "DoS",
    "DoS attacks-SlowHTTPTest":  "DoS",
    "DoS attacks-Slowloris":     "DoS",
    "DDOS attack-HOIC":          "DDoS",
    "DDOS attack-LOIC-UDP":      "DDoS",
    "DDoS attacks-LOIC-HTTP":    "DDoS",
    "SSH-Bruteforce":            "BruteForce",
    "FTP-BruteForce":            "BruteForce",
    "Brute Force -Web":          "WebAttack",
    "Brute Force -XSS":          "WebAttack",
    "SQL Injection":             "WebAttack",
    "Infilteration":             "Infiltration",   # typo in original dataset
    "Bot":                       "Botnet",
}

# ── CICIDS2017 raw labels → unified ──────────────────────────────
# Used in Phase 4 validation only — not during training.
LABEL_MAP_2017 = {
    "BENIGN":                         "BENIGN",
    "DoS Hulk":                       "DoS",
    "DoS GoldenEye":                  "DoS",
    "DoS slowloris":                  "DoS",
    "DoS Slowhttptest":               "DoS",
    "Heartbleed":                     "DoS",        # very few samples, group with DoS
    "DDoS":                           "DDoS",
    "PortScan":                       "PortScan",
    "FTP-Patator":                    "BruteForce",
    "SSH-Patator":                    "BruteForce",
    # 2017 uses a non-standard dash character (\x96) in web attack names
    "Web Attack \x96 Brute Force":    "WebAttack",
    "Web Attack \x96 XSS":           "WebAttack",
    "Web Attack \x96 Sql Injection":  "WebAttack",
    # Handle both the encoded and plain versions
    "Web Attack – Brute Force":       "WebAttack",
    "Web Attack – XSS":              "WebAttack",
    "Web Attack – Sql Injection":     "WebAttack",
    "Infiltration":                   "Infiltration",
    "Bot":                            "Botnet",
}
