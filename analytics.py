import json
import os
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

with open("query_log.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

# 1. Line chart: queries over time
timestamps = [datetime.fromisoformat(log["timestamp"]) for log in logs]
timestamps.sort()
times_per_min = Counter([t.strftime("%Y-%m-%d %H:%M") for t in timestamps])

if "Analytics" not in os.listdir():
    os.mkdir("Analytics")

plt.figure(figsize=(10, 4))
plt.plot(list(times_per_min.keys()), list(times_per_min.values()))
plt.title("Query Frequency Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Analytics/queries_over_time.png")

# 2. Bar chart: queries per user
users = [log["user_id"] for log in logs]
user_counts = Counter(users)

plt.figure(figsize=(6, 4))
plt.bar(user_counts.keys(), user_counts.values())
plt.title("Query Count per User")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Analytics/queries_per_user.png")

# 3. Histogram: answer length distribution
answer_lengths = [len(log["answer"]) for log in logs]

plt.figure(figsize=(6, 4))
plt.hist(answer_lengths, bins=10)
plt.title("Answer Length Distribution")
plt.tight_layout()
plt.savefig("Analytics/answer_length_histogram.png")

print("✅ Charts saved as PNG files.")