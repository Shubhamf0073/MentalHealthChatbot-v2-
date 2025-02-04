import pandas as pd
import json

with open("/Users/shubhamfufal/Library/Mobile Documents/com~apple~CloudDocs/Project/MentalHealthChatbot(v2)/Data/FailedESConv.json") as f:
    data = json.load(f)

rows = []

for entry in data:
    experience = {
        "experience_type": entry["experience_type"],
        "emotion_type": entry["emotion_type"],
        "problem_type": entry["problem_type"],
        "situation": entry["situation"]
    }

    for dialog in entry["dialog"]:
        row = experience.copy()
        row.update({
            "speaker": dialog["speaker"],
            "strategy": dialog["annotation"].get("strategy", ""),
            "feedback": dialog["annotation"].get("feedback", ""),
            "content": dialog["content"]
        })
        rows.append(row)

df = pd.DataFrame(rows)
print(df.dtypes)

df.to_csv("ESConv.csv", index = False)

