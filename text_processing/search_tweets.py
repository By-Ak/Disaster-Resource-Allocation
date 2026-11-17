from datasets import load_dataset

data = load_dataset("QCRI/HumAID-all", verification_mode="no_checks")
train = data["train"]

KEYWORDS = {
    "trapped": ["trapped", "stuck", "under the rubble", "under rubble", "buried"],
    "medical": ["medical", "doctor", "ambulance", "hospital", "medic", "injured"],
    "needs":   ["need help", "please help", "send help", "need food", "need water",
                "need rescue", "sos"],
}

for group, words in KEYWORDS.items():
    hits = [row for row in train
            if any(word in row["tweet_text"].lower() for word in words)]
    print(f"\n=== {group}: {len(hits)} of {len(train)} tweets ===")
    for row in hits[:10]:
        print(f"[{row['class_label']}] {row['tweet_text'][:200]}")