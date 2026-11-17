"""
Text perception: turns disaster reports into numbers per zone.

Two parts work together:
  1. A BERT classifier (trained on HumAID tweets) says what KIND of report each one is
     (injured or dead people, requests or urgent needs, sympathy, ...) and how sure it is.
     It is used to throw away reports that are not about needs, and to measure confidence.
  2. A rule-based extractor reads each report and finds:
       - the ZONE it talks about
       - the COUNTS: people trapped, people injured, people needing food/water
       - the URGENCY of the wording
     (HumAID has no counts or zones, so these are NOT learned. They are rules.)

Then the reports are combined into one state per zone.

Usage:
  python text_perception.py demo                               (no BERT needed, runs anywhere)
  python text_perception.py train --sample 2000 --epochs 1     (quick test of training)
  python text_perception.py train --epochs 2                   (full training, needs a GPU)
  python text_perception.py predict --model bert_humaid --file my_reports.txt
"""
import argparse
import re

# ====================== settings you can edit ======================
ZONE_NAMES = ["Zone A", "Zone B", "Zone C", "Zone D"]
# A report is assigned to a zone when it mentions one of these words (or a word starting with them).
ZONE_KEYWORDS = {
    0: ["north", "downtown"],
    1: ["west", "hillside"],
    2: ["east", "riverside"],
    3: ["south", "harbor", "harbour"],
}

# Words that signal each kind of need.
NEED_KEYWORDS = {
    "trapped": ["trapped", "stuck", "stranded", "cut off", "rooftop", "rooftops",
                "need rescue", "needs rescue", "require rescue", "send rescue",
                "waiting for rescue", "awaiting rescue"],
    "injured": ["injured", "hurt", "wounded", "casualties", "bleeding",
                "need medical", "needs medical", "need doctors", "medics", "ambulance"],
    "food_need": ["food", "water", "hungry", "starving", "starve", "thirsty",
                  "supplies", "rations"],
}

# Words that raise urgency. Each distinct word adds 0.25 (max 1.0).
URGENT_WORDS = ["urgent", "urgently", "immediately", "asap", "emergency", "critical",
                "sos", "please help", "dying", "running out", "life-threatening", "hurry"]

# If a report names a need but gives no number, assume this many people (an assumption).
DEFAULT_COUNT_IF_UNSTATED = 3.0

# Vague quantities turned into numbers (assumptions).
VAGUE_COUNTS = {"a couple": 2, "a few": 3, "few": 3, "some": 4, "several": 5,
                "many": 10, "dozens": 24, "hundreds": 200}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
                "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}

# HumAID categories (matched by their start, so small spelling differences are fine).
IRRELEVANT_PREFIXES = ("not_humanitarian", "sympathy", "other_relevant", "dont_know",
                       "don't_know", "caution", "rescue_volunteering")
URGENT_PREFIXES = ("requests", "injured", "missing", "displaced")

# Biggest count we accept for one need in one zone. Matches the simulator's limit (60 people per
# zone); a bigger number in a tweet is almost always a country-wide total from the news.
MAX_PEOPLE_PER_ZONE = 60

# Phrases that make a tweet read like a news report instead of a report from the area.
# Such tweets are skipped. Edit this list freely.
NEWS_PHRASES = ["death toll", "according to", "officials said", "official said", "reportedly",
                "breaking", "latest:", "update:", "reuters", "bbc", "cnn",
                "authorities said", "ministry", "government says",
                "toll rises", "toll has risen", "toll now"]

MAX_TOKENS = 64          # tweets are short
HUMAID_DATASET = "QCRI/HumAID-all"   # columns: tweet_text, class_label


# ====================== rule-based extraction ======================
_NUMBER_TOKEN = (r"(?:\d[\d,]*|" +
                 "|".join(sorted(list(NUMBER_WORDS) + list(VAGUE_COUNTS), key=len, reverse=True)) + r")")


def _keyword_group(words):
    return r"(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")"


# "<number> up to 3 words <keyword>", for example "40 people trapped" or "12 injured"
_COUNT_PATTERNS = {
    need: re.compile(rf"\b({_NUMBER_TOKEN})(?!\w)(?:\s+[\w'-]+){{0,3}}?\s+{_keyword_group(words)}\b",
                     re.IGNORECASE)
    for need, words in NEED_KEYWORDS.items()
}
_PRESENCE_PATTERNS = {
    need: re.compile(rf"\b{_keyword_group(words)}\b", re.IGNORECASE)
    for need, words in NEED_KEYWORDS.items()
}
_ZONE_PATTERNS = {
    zone: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\w*", re.IGNORECASE)
    for zone, words in ZONE_KEYWORDS.items()
}


def _to_number(token):
    token = token.lower().replace(",", "")
    if token.isdigit():
        return float(token)
    if token in VAGUE_COUNTS:
        return float(VAGUE_COUNTS[token])
    return float(NUMBER_WORDS[token])


def find_zones(text):
    """Zones mentioned in the report."""
    return [zone for zone, pattern in _ZONE_PATTERNS.items() if pattern.search(text)]


def urgency_from_words(text):
    lowered = text.lower()
    score = 0.25 * sum(1 for word in URGENT_WORDS if word in lowered)
    if "!" in text:
        score += 0.1
    return min(1.0, score)


def looks_like_news(text):
    """True when the tweet reads like a news report (death toll, 'according to', and so on)."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in NEWS_PHRASES)


def extract_from_report(text):
    """Read one report. Returns the zones it mentions, the counts it states, and its urgency."""
    counts = {}
    for need in NEED_KEYWORDS:
        numbers = [_to_number(match.group(1)) for match in _COUNT_PATTERNS[need].finditer(text)]
        if numbers:
            counts[need] = max(numbers)
        elif _PRESENCE_PATTERNS[need].search(text):
            counts[need] = DEFAULT_COUNT_IF_UNSTATED
    counts = {need: min(count, MAX_PEOPLE_PER_ZONE) for need, count in counts.items()}
    return {"zones": find_zones(text), "counts": counts, "urgency": urgency_from_words(text)}


# ====================== BERT category model ======================
def _starts_with_any(label, prefixes):
    label = label.lower().replace(" ", "_")
    return any(label.startswith(prefix) for prefix in prefixes)


class BertCategoryModel:
    """Loads the trained BERT and gives (category, confidence, all probabilities) per report."""

    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()

    def predict(self, texts, batch_size=32):
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, truncation=True, padding=True, max_length=MAX_TOKENS,
                                     return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
            id_to_label = self.model.config.id2label
            for row in probabilities:
                best = int(row.argmax())
                results.append((id_to_label[best], float(row[best]),
                                {id_to_label[i]: float(p) for i, p in enumerate(row)}))
        return results


# ====================== combine reports into one state per zone ======================
def estimate_zone_state(reports, category_model=None):
    """reports: list of strings. category_model: optional BertCategoryModel.
    Returns (state per zone, how many reports were skipped and why).

    For each zone: trapped, injured, food_need (people), urgency (0-1), num_reports,
    text_confidence (0-1). When several reports talk about the same zone and the same need,
    the LARGEST count is kept, because tweets about one incident repeat each other and
    adding them up would count the same people twice."""
    predictions = category_model.predict(reports) if category_model else [None] * len(reports)
    state = {zone: {"trapped": 0.0, "injured": 0.0, "food_need": 0.0, "urgency": 0.0,
                    "num_reports": 0, "confidences": []} for zone in range(len(ZONE_NAMES))}
    skipped = {"not_relevant": 0, "news": 0, "no_zone": 0, "ambiguous_zone": 0, "no_need": 0}

    for text, prediction in zip(reports, predictions):
        if looks_like_news(text):
            skipped["news"] += 1
            continue
        category, confidence, probabilities = prediction if prediction else (None, 1.0, {})
        if category is not None and _starts_with_any(category, IRRELEVANT_PREFIXES):
            skipped["not_relevant"] += 1
            continue
        info = extract_from_report(text)
        if not info["zones"]:
            skipped["no_zone"] += 1
            continue
        if len(info["zones"]) > 1:
            skipped["ambiguous_zone"] += 1
            continue
        if not info["counts"]:
            skipped["no_need"] += 1
            continue

        zone = info["zones"][0]
        for need, count in info["counts"].items():
            state[zone][need] = max(state[zone][need], count)
        bert_urgency = sum(p for label, p in probabilities.items()
                           if _starts_with_any(label, URGENT_PREFIXES))
        urgency = min(1.0, info["urgency"] + 0.5 * bert_urgency)
        state[zone]["urgency"] = max(state[zone]["urgency"], urgency)
        state[zone]["num_reports"] += 1
        state[zone]["confidences"].append(confidence)

    for zone_state in state.values():
        confidences = zone_state.pop("confidences")
        reports_factor = min(1.0, zone_state["num_reports"] / 3)     # 3+ reports = fully trusted
        zone_state["text_confidence"] = (reports_factor * sum(confidences) / len(confidences)
                                         if confidences else 0.0)
    return state, skipped


def print_state(state, skipped):
    print(f"{'Zone':8s}{'Trapped':>9s}{'Injured':>9s}{'Food need':>11s}{'Urgency':>9s}{'Reports':>9s}{'Confidence':>12s}")
    for zone, s in state.items():
        print(f"{ZONE_NAMES[zone]:8s}{s['trapped']:9.0f}{s['injured']:9.0f}{s['food_need']:11.0f}"
              f"{s['urgency']:9.2f}{s['num_reports']:9d}{s['text_confidence']:12.2f}")
    print("Skipped reports:", skipped)


# ====================== training the BERT classifier ======================
def train_classifier(output_dir, epochs, base_model, batch_size, sample):
    import inspect
    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    print("CUDA GPU available:", torch.cuda.is_available(),
          "(training on CPU takes hours, use --sample for a quick test)" if not torch.cuda.is_available() else "")
    # The dataset's info lists a "dev" split that the downloaded files do not contain, which makes
    # the library's default split check fail (ExpectedMoreSplitsError), so that check is skipped.
    data = load_dataset(HUMAID_DATASET, verification_mode="no_checks")
    validation_name = "validation" if "validation" in data else "dev"
    print("Splits found:", {name: len(split) for name, split in data.items()})
    feature = data["train"].features["class_label"]
    if hasattr(feature, "names"):                       # labels already stored as numbers
        label_names, label_to_id = list(feature.names), None
    else:                                               # labels stored as text
        label_names = sorted(set(data["train"]["class_label"]))
        label_to_id = {name: i for i, name in enumerate(label_names)}
    print("Categories:", label_names)

    if sample:
        data["train"] = data["train"].shuffle(seed=0).select(range(min(sample, len(data["train"]))))

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def prepare(batch):
        encoded = tokenizer(batch["tweet_text"], truncation=True, max_length=MAX_TOKENS)
        encoded["labels"] = (batch["class_label"] if label_to_id is None
                             else [label_to_id[name] for name in batch["class_label"]])
        return encoded

    tokenized = data.map(prepare, batched=True, remove_columns=data["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(label_names),
        id2label=dict(enumerate(label_names)), label2id={n: i for i, n in enumerate(label_names)})

    def metrics(eval_prediction):
        predicted = np.argmax(eval_prediction.predictions, axis=-1)
        return {"accuracy": accuracy_score(eval_prediction.label_ids, predicted),
                "macro_f1": f1_score(eval_prediction.label_ids, predicted, average="macro")}

    # this argument was renamed between library versions
    eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters \
        else "evaluation_strategy"
    arguments = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, learning_rate=2e-5, weight_decay=0.01,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(), save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", report_to="none", **{eval_key: "epoch"})
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized["train"],
                      eval_dataset=tokenized[validation_name], data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=metrics)
    trainer.train()

    test = trainer.predict(tokenized["test"])
    predicted = np.argmax(test.predictions, axis=-1)
    report = classification_report(test.label_ids, predicted, target_names=label_names, digits=3)
    print("\nTEST SET RESULTS (use these in your report)\n" + report)
    with open("bert_test_report.txt", "w") as report_file:
        report_file.write(report)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}/ and the test report to bert_test_report.txt")


# ====================== demo and command line ======================
DEMO_REPORTS = [
    "About 40 people trapped on rooftops in the Eastern Area, please send rescue boats urgently!",
    "12 injured near the riverside clinic, ambulance needed immediately",
    "Northern district: about 30 people need food and drinking water.",
    "Several families stranded in the west side, roads cut off",
    "Thinking of everyone affected by the flood, stay strong",
    "Dozens injured in the south after the bridge collapsed",
    "Eastern Area: 15 people trapped, 5 injured. Need rescue and medics ASAP",
    "Food supplies running out in the south, 20 hungry people",
    "At least 10 people hurt somewhere, no idea where",
]
# what the extractor should produce per zone for the demo reports above
DEMO_EXPECTED = {0: {"food_need": 30}, 1: {"trapped": 5},
                 2: {"trapped": 40, "injured": 12}, 3: {"injured": 24, "food_need": 20}}


def run_demo():
    print("Demo reports:")
    for report in DEMO_REPORTS:
        info = extract_from_report(report)
        zones = [ZONE_NAMES[z] for z in info["zones"]] or ["no zone"]
        print(f"  - {report}\n      -> {zones}, counts {info['counts']}, urgency {info['urgency']:.2f}")
    state, skipped = estimate_zone_state(DEMO_REPORTS)
    print("\nCombined state per zone:")
    print_state(state, skipped)

    wrong = 0
    for zone, expected in DEMO_EXPECTED.items():
        for need in ("trapped", "injured", "food_need"):
            if state[zone][need] != expected.get(need, 0):
                wrong += 1
                print(f"MISMATCH {ZONE_NAMES[zone]} {need}: got {state[zone][need]}, expected {expected.get(need, 0)}")
    print("\nAll demo numbers match." if wrong == 0 else f"\n{wrong} numbers differ from what was expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "train", "predict"])
    parser.add_argument("--model", default="bert_humaid", help="folder with the trained BERT")
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample", type=int, default=0, help="train on only this many tweets (quick test)")
    parser.add_argument("--file", default=None, help="text file with one report per line")
    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "train":
        train_classifier(args.model, args.epochs, args.base_model, args.batch_size, args.sample)
    else:
        with open(args.file, encoding="utf-8") as report_file:
            reports = [line.strip() for line in report_file if line.strip()]
        zone_state, skipped_reports = estimate_zone_state(reports, BertCategoryModel(args.model))
        print_state(zone_state, skipped_reports)