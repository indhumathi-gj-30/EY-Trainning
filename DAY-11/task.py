import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "indhu_ppg_20260611T074737_len148s.csv"

ppg_data = pd.read_csv(DATA_PATH)
ppg_data = ppg_data.sort_values("timestamp_ms").drop_duplicates(subset=["timestamp_ms"]).reset_index(drop=True)

ppg_data["duration_s"] = (ppg_data["timestamp_ms"] - ppg_data["timestamp_ms"].iloc[0]) / 1000.0

ppg_data["red_norm"] = (ppg_data["red"] - ppg_data["red"].mean()) / ppg_data["red"].std(ddof=0)
ppg_data["ir_norm"] = (ppg_data["ir"] - ppg_data["ir"].mean()) / ppg_data["ir"].std(ddof=0)


plt.figure(figsize=(14, 6))
plt.plot(ppg_data["duration_s"], ppg_data["red_norm"], color="crimson", label="Normalized Red Signal", alpha=0.75)
plt.plot(ppg_data["duration_s"], ppg_data["ir_norm"], color="darkblue", label="Normalized IR Signal", alpha=0.75)

plt.title("Comparison of Standardized Red and IR Channels")
plt.xlabel("Duration (s)")
plt.ylabel("Z-Score Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("red_ir_overlay_standardized.png", dpi=150, bbox_inches="tight")
plt.show()

ppg_data[["duration_s", "red_norm", "ir_norm"]].to_csv("standardized_output.csv", index=False)
print("Saved: red_ir_overlay_standardized.png")
print("Saved: standardized_output.csv")