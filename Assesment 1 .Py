
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

geyser_df = sns.load_dataset('geyser')

print(geyser_df.shape, "shape")

# Head (first 5 rows)
print("\nHead:", geyser_df.head())

# Info (structure of data)
print("\nInfo:")
print(geyser_df.info())

# Describe (statistical summary)
print("\nDescribe:", geyser_df.describe())

# Missing value handling
geyser_df['duration'] = geyser_df['duration'].fillna(geyser_df['duration'].mean())
geyser_df['waiting'] = geyser_df['waiting'].fillna(geyser_df['waiting'].mean())

print("\nAfter cleaning:", geyser_df.isnull().sum())

# Feature engineering
geyser_df['eruption_label'] = geyser_df['duration'].apply(
    lambda value: 'Short' if value < 3 else 'Long'
)

print("\nNew Feature Added:", geyser_df[['duration', 'eruption_label']].head())

# NumPy arrays
duration_values = np.array(geyser_df['duration'])
waiting_values = np.array(geyser_df['waiting'])

mean_duration_value = np.mean(duration_values)
print("Mean duration (NumPy):", mean_duration_value)

max_waiting_value = np.max(waiting_values)
print("Max waiting (NumPy):", max_waiting_value)

# Create figure
figure, axes_grid = plt.subplots(2, 2, figsize=(14, 10))
figure.suptitle("Geyser Data Visualization", fontsize=16, fontweight='bold')

# 1. Histogram
axis = axes_grid[0, 0]
axis.hist(geyser_df['duration'], bins=20, color='skyblue')
axis.set_title("1. Distribution of Duration")
axis.set_xlabel("Duration")
axis.set_ylabel("Frequency")

axis.annotate(
    "Peak",
    xy=(geyser_df['duration'].mean(), 20),
    xytext=(4, 30),
    arrowprops=dict(facecolor='black')
)

# 2. Scatter Plot
axis = axes_grid[0, 1]
axis.scatter(geyser_df['duration'], geyser_df['waiting'], color='green', alpha=0.6)
axis.set_title("2. Duration vs Waiting Time")
axis.set_xlabel("Duration")
axis.set_ylabel("Waiting")

axis.annotate(
    "Trend",
    xy=(4, 80),
    xytext=(2, 90),
    arrowprops=dict(facecolor='black')
)

# 3. Box Plot
axis = axes_grid[1, 0]
sns.boxplot(x='kind', y='duration', data=geyser_df, ax=axis)
axis.set_title("3. Duration by Eruption Type")
axis.set_xlabel("Eruption Type")
axis.set_ylabel("Duration")

axis.annotate(
    "Short vs Long",
    xy=(0, 2),
    xytext=(0.5, 4.5),
    arrowprops=dict(facecolor='black')
)

# 4. Count Plot
axis = axes_grid[1, 1]
sns.countplot(x='kind', hue='kind', data=geyser_df, palette='Set2', ax=axis, legend=False)
axis.set_title("4. Count of Eruption Type")
axis.set_xlabel("Eruption Type")
axis.set_ylabel("Count")

counts_series = geyser_df['kind'].value_counts()

axis.annotate(
    f"Max: {counts_series.max()}",
    xy=(0, counts_series.max()),
    xytext=(0.5, counts_series.max() + 10),
    arrowprops=dict(facecolor='black')
)

plt.tight_layout()
plt.show()

# Key Insights
print("\nKey Insights:")

print("1. The histogram shows a bimodal distribution of eruption duration, indicating two types: short and long eruptions.")
print("2. The scatter plot shows a positive relationship between duration and waiting time.")
print("3. The boxplot highlights clear separation between short and long eruptions.")