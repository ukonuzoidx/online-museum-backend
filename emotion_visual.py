import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your exported CSV file (update the filename if different)
df = pd.read_csv("emotion_log.csv")

# Step 1: Clean and preprocess the data
# Drop rows with missing or zero confidence
df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')  # Ensure numeric
df_clean = df.dropna(subset=['Confidence'])  # Drop NaNs
df_clean = df_clean[df_clean['Confidence'] > 0]  # Remove zero values

# Optional: Strip whitespace or standardise emotion labels
df_clean['Emotion'] = df_clean['Emotion'].str.strip().str.capitalize()

# Step 2: Bar Chart – Emotion Distribution
emotion_counts = df_clean['Emotion'].value_counts()
plt.figure(figsize=(8, 8))
# emotion_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
# bar chart instead of pie chart
plt.bar(emotion_counts.index, emotion_counts.values, color=sns.color_palette("pastel"))
plt.title("Figure 5.3.3.1 – Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("emotion_distribution.png")
plt.show()




# Step 3: Histogram – Confidence Scores Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Confidence'], bins=20, kde=True, color='skyblue')
plt.title("Figure 5.3.3.2 – Distribution of Confidence Scores")
plt.xlabel("Confidence (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("confidence_score_histogram.png")
plt.show()

# Step 4: Bar Chart – Average Confidence per Emotion
avg_conf = df_clean.groupby('Emotion')['Confidence'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_conf.index, y=avg_conf.values, palette="viridis")
plt.title("Figure 5.3.3.3 – Average Confidence by Emotion")
plt.xlabel("Emotion")
plt.ylabel("Average Confidence (%)")
plt.tight_layout()
plt.savefig("average_confidence_per_emotion.png")
plt.show()
