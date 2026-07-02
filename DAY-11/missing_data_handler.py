import pandas as pd
import numpy as np
DATA_SOURCE_PATH = "Indhu_ppg_20260611T074737_len148s.csv"

# Ingest raw dataset
raw_dataframe = pd.read_csv(DATA_SOURCE_PATH)
null_ratios = (raw_dataframe.isnull().sum() / len(raw_dataframe)) * 100
null_summary_df = pd.DataFrame({
    "Data_Field": raw_dataframe.columns,
    "Null_Total_Count": raw_dataframe.isnull().sum().values,
    "Null_Percentage_Share": null_ratios.values
})

print("Comprehensive Null Values Matrix:")
print(null_summary_df)
raw_dataframe = raw_dataframe.sort_values("timestamp_ms").drop_duplicates(subset=["timestamp_ms"]).reset_index(drop=True)
clean_signal_df = raw_dataframe.copy()

target_channels = ["red", "ir", "red_corrected", "ir_corrected"]

# Execute linear numerical interpolation along the timeline axis
clean_signal_df[target_channels] = clean_signal_df[target_channels].interpolate(method="linear")

# Resolve leftover edge boundary null entries
clean_signal_df[target_channels] = clean_signal_df[target_channels].bfill().ffill()

print("\nVerification of Post-Imputation Null State:")
print(clean_signal_df[target_channels].isnull().sum())