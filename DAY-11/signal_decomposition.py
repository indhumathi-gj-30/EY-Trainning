import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
DATA_SOURCE = "Indhu_ppg_20260611T074737_len148s.csv"

# Load stream and drop chronologically conflicting duplicate frames
pulse_df = pd.read_csv(DATA_SOURCE)
pulse_df = pulse_df.sort_values("timestamp_ms").drop_duplicates(subset=["timestamp_ms"]).reset_index(drop=True)
pulse_df["delta_time_ms"] = pulse_df["timestamp_ms"] - pulse_df["timestamp_ms"].iloc[0]
pulse_df["temporal_axis"] = pd.to_datetime(pulse_df["delta_time_ms"], unit="ms")
pulse_df = pulse_df.set_index("temporal_axis")


resampled_pulse = pulse_df.resample("20ms").mean(numeric_only=True)
resampled_pulse["ir_corrected"] = resampled_pulse["ir_corrected"].interpolate(method="linear").bfill().ffill()

raw_signal_vector = resampled_pulse["ir_corrected"].dropna().astype(float).values
SAMPLING_RATE_HZ = 50   

zero_centered_vector = raw_signal_vector - np.mean(raw_signal_vector)
spectrum_freqs, spectrum_power = periodogram(zero_centered_vector, fs=SAMPLING_RATE_HZ)


frequency_passband = (spectrum_freqs >= 0.5) & (spectrum_freqs <= 3.0)

peak_signal_freq = spectrum_freqs[frequency_passband][np.argmax(spectrum_power[frequency_passband])]
calculated_sample_period = int(round(SAMPLING_RATE_HZ / peak_signal_freq))

print("Calculated Sample Window Period:", calculated_sample_period)

signal_series_obj = pd.Series(raw_signal_vector)

decomp_output = seasonal_decompose(
    signal_series_obj,
    model="additive",
    period=calculated_sample_period,
    extrapolate_trend="freq"
)

fig, signal_plots = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
signal_plots[0].plot(decomp_output.observed, color="darkcyan")
signal_plots[0].set_title("Input Raw Signal (Observed)")

signal_plots[1].plot(decomp_output.trend, color="crimson")
signal_plots[1].set_title("Baseline Underlying Trend")

signal_plots[2].plot(decomp_output.seasonal, color="royalblue")
signal_plots[2].set_title("Cyclic Seasonal Patterns")

signal_plots[3].plot(decomp_output.resid, color="darkslate grey")
signal_plots[3].set_title("Stochastic Residual Components")

plt.tight_layout()
plt.show()