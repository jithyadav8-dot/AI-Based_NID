# inspect_alerts.py
# Run while Phase 7 is running — fetches and analyses the stored alerts

import json
import urllib.request
from collections import defaultdict, Counter
from datetime import datetime

BASE = "http://localhost:8000"

try:
    # ── Fetch all alerts ──────────────────────────────────────────────
    with urllib.request.urlopen(f"{BASE}/alerts?limit=1000") as r:
        alerts = json.loads(r.read())

    with urllib.request.urlopen(f"{BASE}/stats") as r:
        stats = json.loads(r.read())
except Exception as e:
    print(f"Error connecting to {BASE}: {e}")
    exit(1)

print(f"\n{'='*60}")
print(f"ALERT STORE ANALYSIS  ({len(alerts)} non-NORMAL alerts)")
print(f"{'='*60}")
print(f"\nUptime     : {stats['uptime_seconds']:.0f}s "
      f"({stats['uptime_seconds']/60:.1f} min)")
print(f"Scored     : {stats['windows_scored']:,} windows")
print(f"Latency    : {stats['avg_latency_ms']}ms avg")
print(f"Active flows: {stats['active_flows']}")

# ── Level breakdown ───────────────────────────────────────────────
print(f"\n--- Alert level breakdown ---")
level_counts = Counter(a.get("level") for a in alerts)
total_scored = stats.get("windows_scored", 0)
for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    n   = level_counts.get(level, 0)
    pct = n / total_scored * 100 if total_scored else 0
    bar = "#" * int(pct * 2)
    print(f"  {level:<10} {n:>5}  ({pct:.2f}% of scored)  {bar}")

# ── Label breakdown ───────────────────────────────────────────────
print(f"\n--- Predicted labels ---")
label_counts = Counter(a.get("label") for a in alerts)
for label, count in label_counts.most_common():
    bar = "#" * int(count / len(alerts) * 30) if alerts else ""
    print(f"  {label:<25} {count:>4}  {bar}")

# ── XGBoost vs AE breakdown ───────────────────────────────────────
print(f"\n--- Detection source ---")
xgb_only = sum(1 for a in alerts
               if a.get("xgb_is_attack") and not a.get("ae_is_anomaly"))
ae_only  = sum(1 for a in alerts
               if not a.get("xgb_is_attack") and a.get("ae_is_anomaly"))
both     = sum(1 for a in alerts
               if a.get("xgb_is_attack") and a.get("ae_is_anomaly"))
neither  = sum(1 for a in alerts
               if not a.get("xgb_is_attack") and not a.get("ae_is_anomaly"))

print(f"  XGBoost only (known attack)    : {xgb_only:>4}")
print(f"  AE only (anomaly, XGB=BENIGN)  : {ae_only:>4}")
print(f"  Both agree (CRITICAL candidate): {both:>4}")
print(f"  Neither (LOW from AE score)    : {neither:>4}")

# ── TOP 10 most flagged flows ─────────────────────────────────────
print(f"\n--- Top 10 most flagged flows ---")
flow_counts = Counter(
    f"{a.get('src_ip')}:{a.get('src_port')} -> {a.get('dst_ip')}:{a.get('dst_port')}"
    for a in alerts
)
for flow, count in flow_counts.most_common(10):
    print(f"  {count:>4}x  {flow}")

# ── HIGH and CRITICAL alerts ──────────────────────────────────────
high_crit = [a for a in alerts if a.get("level") in ("HIGH", "CRITICAL")]
if high_crit:
    print(f"\n--- HIGH / CRITICAL alerts ({len(high_crit)}) ---")
    for a in high_crit[:20]:     # show up to 20
        ts  = datetime.fromtimestamp(a.get("timestamp", 0)).strftime("%H:%M:%S")
        print(f"  [{ts}] {a.get('level', ''):<10} {a.get('label', ''):<20} "
              f"{a.get('src_ip')}:{a.get('src_port')} -> "
              f"{a.get('dst_ip')}:{a.get('dst_port')} "
              f"({a.get('protocol')})")
        print(f"           XGB={a.get('xgb_class')} ({a.get('xgb_confidence', 0):.3f})  "
              f"AE={a.get('ae_error', 0):.4f}  severity={a.get('severity')}")
else:
    print(f"\n  No HIGH/CRITICAL alerts in store.")

# ── AE error distribution for MEDIUM alerts ───────────────────────
medium = [a for a in alerts
          if a.get("level") == "MEDIUM" and not a.get("xgb_is_attack")]
if medium:
    import statistics
    ae_errs = [a.get("ae_error", 0) for a in medium]
    print(f"\n--- AE-only MEDIUM alerts: error distribution ---")
    print(f"  Count  : {len(ae_errs)}")
    print(f"  Mean   : {statistics.mean(ae_errs):.4f}")
    print(f"  Median : {statistics.median(ae_errs):.4f}")
    print(f"  Max    : {max(ae_errs):.4f}")
    if alerts and "ae_threshold" in alerts[0]:
        print(f"  Threshold: {alerts[0]['ae_threshold']:.4f}")

    # Bucket by error magnitude
    mild     = sum(1 for e in ae_errs if e < 5)
    moderate = sum(1 for e in ae_errs if 5 <= e < 15)
    high     = sum(1 for e in ae_errs if e >= 15)
    print(f"\n  Error < 5   (mild)     : {mild}")
    print(f"  Error 5-15  (moderate) : {moderate}")
    print(f"  Error >= 15 (high)     : {high}")

print(f"\n{'='*60}")
print("Done.")
