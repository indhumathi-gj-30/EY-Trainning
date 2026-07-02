import pandas as pd

DATA_SET_PATH = "Indhu_ppg_20260611T074737_len148s.csv"
signal_data = pd.read_csv(DATA_SET_PATH)
signal_data = signal_data.drop(columns=["seq", "timestamp_ms"])
signal_data["red_moving_avg"] = signal_data["red"].rolling(window=50, min_periods=1).mean()
signal_data["ir_moving_avg"] = signal_data["ir"].rolling(window=50, min_periods=1).mean()
signal_data["red_residual_diff"] = signal_data["red"] - signal_data["red_moving_avg"]
signal_data["ir_residual_diff"] = signal_data["ir"] - signal_data["ir_moving_avg"]
signal_data["red_scaled_bound"] = (signal_data["red"] - signal_data["red"].min()) / (signal_data["red"].max() - signal_data["red"].min())
signal_data["ir_scaled_bound"] = (signal_data["ir"] - signal_data["ir"].min()) / (signal_data["ir"].max() - signal_data["ir"].min())

print("Extracted Feature Engineering Columns:")
print(signal_data.columns.tolist())
print("\nSample Preview (First 5 Rows):")
print(signal_data.head())