# check_2017_columns.py
# Run this once to find the source port column name in 2017

import pandas as pd
import glob

f = sorted(glob.glob("raw/cic2017/MachineLearningCVE/*.csv"))[0]
df = pd.read_csv(f, nrows=5, low_memory=False, encoding_errors="replace")
df.columns = df.columns.str.strip()

# Look for anything port-related
port_cols = [c for c in df.columns if "port" in c.lower() or "Port" in c]
print("Port-related columns in 2017:")
for c in port_cols:
    print(f"  '{c}'")

print("\nAll 2017 columns:")
for c in sorted(df.columns):
    print(f"  '{c}'")
