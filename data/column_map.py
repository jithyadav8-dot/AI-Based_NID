# Renames CSE-CIC-IDS2018 column names to the CICIDS2017 canonical scheme.
# After this renaming both datasets share identical column names.
# Columns not in this map are already identical between the two datasets.

COLUMN_MAP_2018_TO_CANONICAL = {
    "Dst Port":          "Destination Port",
    "Tot Fwd Pkts":      "Total Fwd Packets",
    "Tot Bwd Pkts":      "Total Backward Packets",
    "TotLen Fwd Pkts":   "Total Length of Fwd Packets",
    "TotLen Bwd Pkts":   "Total Length of Bwd Packets",
    "Fwd Pkt Len Max":   "Fwd Packet Length Max",
    "Fwd Pkt Len Min":   "Fwd Packet Length Min",
    "Fwd Pkt Len Mean":  "Fwd Packet Length Mean",
    "Fwd Pkt Len Std":   "Fwd Packet Length Std",
    "Bwd Pkt Len Max":   "Bwd Packet Length Max",
    "Bwd Pkt Len Min":   "Bwd Packet Length Min",
    "Bwd Pkt Len Mean":  "Bwd Packet Length Mean",
    "Bwd Pkt Len Std":   "Bwd Packet Length Std",
    "Flow Byts/s":       "Flow Bytes/s",
    "Flow Pkts/s":       "Flow Packets/s",
    "Fwd IAT Tot":       "Fwd IAT Total",
    "Bwd IAT Tot":       "Bwd IAT Total",
    "Fwd Header Len":    "Fwd Header Length",
    "Bwd Header Len":    "Bwd Header Length",
    "Fwd Pkts/s":        "Fwd Packets/s",
    "Bwd Pkts/s":        "Bwd Packets/s",
    "Pkt Len Min":       "Min Packet Length",
    "Pkt Len Max":       "Max Packet Length",
    "Pkt Len Mean":      "Packet Length Mean",
    "Pkt Len Std":       "Packet Length Std",
    "Pkt Len Var":       "Packet Length Variance",
    "FIN Flag Cnt":      "FIN Flag Count",
    "SYN Flag Cnt":      "SYN Flag Count",
    "RST Flag Cnt":      "RST Flag Count",
    "PSH Flag Cnt":      "PSH Flag Count",
    "ACK Flag Cnt":      "ACK Flag Count",
    "URG Flag Cnt":      "URG Flag Count",
    "ECE Flag Cnt":      "ECE Flag Count",
    "Pkt Size Avg":      "Average Packet Size",
    "Fwd Seg Size Avg":  "Avg Fwd Segment Size",
    "Bwd Seg Size Avg":  "Avg Bwd Segment Size",
    "Fwd Byts/b Avg":    "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg":    "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg":  "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg":    "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg":    "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg":  "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts":  "Subflow Fwd Packets",
    "Subflow Fwd Byts":  "Subflow Fwd Bytes",
    "Subflow Bwd Pkts":  "Subflow Bwd Packets",
    "Subflow Bwd Byts":  "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min":  "min_seg_size_forward",
}


def apply_2018_column_map(df):
    """Strip whitespace and rename 2018 columns to canonical names."""
    df.columns = df.columns.str.strip()
    return df.rename(columns=COLUMN_MAP_2018_TO_CANONICAL)


def apply_2017_column_map(df):
    """2017 column names are already canonical — just strip whitespace."""
    df.columns = df.columns.str.strip()
    return df
